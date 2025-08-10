# Multi-stage build for optimized image size and security
# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib (required for technical indicators)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# Copy requirements file
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    libgomp1 \
    curl \
    netcat-openbsd \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib from builder
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib /usr/include/ta-lib

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user with specific UID/GID
RUN groupadd -r -g 1000 crypto && \
    useradd -r -u 1000 -g crypto -d /home/crypto -s /bin/bash crypto && \
    mkdir -p /home/crypto && \
    chown -R crypto:crypto /home/crypto

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code with correct ownership
COPY --chown=crypto:crypto . /app

# Create necessary directories with correct permissions
RUN mkdir -p \
    /app/data/models \
    /app/data/features \
    /app/data/backtest \
    /app/data/cache \
    /app/logs \
    /app/temp \
    && chown -R crypto:crypto /app \
    && chmod -R 755 /app/data \
    && chmod -R 755 /app/logs

# Create volume mount points
VOLUME ["/app/data", "/app/logs"]

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for PostgreSQL\n\
if [ -n "$POSTGRES_HOST" ]; then\n\
    echo "Waiting for PostgreSQL..."\n\
    while ! nc -z "$POSTGRES_HOST" "${POSTGRES_PORT:-5432}"; do\n\
        sleep 1\n\
    done\n\
    echo "PostgreSQL is ready"\n\
fi\n\
\n\
# Wait for Redis\n\
if [ -n "$REDIS_HOST" ]; then\n\
    echo "Waiting for Redis..."\n\
    while ! nc -z "$REDIS_HOST" "${REDIS_PORT:-6379}"; do\n\
        sleep 1\n\
    done\n\
    echo "Redis is ready"\n\
fi\n\
\n\
# Execute command\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Switch to non-root user
USER crypto

# Expose ports
# 8000 - FastAPI
# 9090 - Prometheus metrics
# 8888 - Jupyter (optional)
EXPOSE 8000 9090 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command - can be overridden
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop"]