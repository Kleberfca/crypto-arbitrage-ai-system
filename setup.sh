#!/bin/bash

# Crypto Arbitrage AI System - Setup Script
# Automated setup for development environment
# Author: Crypto Arbitrage AI Team
# Date: 2025

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║     🚀 CRYPTO ARBITRAGE AI SYSTEM SETUP 🚀          ║"
echo "║     Professional Cryptocurrency Arbitrage Bot        ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print colored messages
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
    print_success "Python 3.11 found"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]]; then
        PYTHON_CMD=python3
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.11+ required (found $PYTHON_VERSION)"
        exit 1
    fi
else
    print_error "Python 3.11+ not found"
    exit 1
fi

# Check Docker
print_status "Checking Docker..."
if command -v docker &> /dev/null; then
    print_success "Docker found"
else
    print_warning "Docker not found - please install Docker"
fi

# Check Docker Compose
print_status "Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    print_success "Docker Compose found"
else
    print_warning "Docker Compose not found - please install Docker Compose"
fi

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p data/{models,features,backtest,logs}
mkdir -p logs
mkdir -p notebooks
mkdir -p deployment/monitoring/{dashboards,datasources}
mkdir -p scripts
mkdir -p tests/{unit,integration,performance}
print_success "Directories created"

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "Pip upgraded"

# Install dependencies
print_status "Installing Python dependencies (this may take a few minutes)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Setup environment file
print_status "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success "Created .env file from template"
        print_warning "Please edit .env and add your API keys"
    else
        print_error ".env.example not found"
    fi
else
    print_warning ".env file already exists"
fi

# Initialize log files
print_status "Initializing log files..."
touch logs/arbitrage.log
touch logs/trades.jsonl
print_success "Log files initialized"

# Create sample data directory
print_status "Creating sample data directory..."
mkdir -p data/sample
print_success "Sample data directory created"

# Start Docker services (optional)
read -p "Do you want to start Docker services now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting Docker services..."
    
    # Start only essential services
    docker-compose up -d redis postgres
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check if services are running
    if docker ps | grep -q redis; then
        print_success "Redis is running"
    else
        print_warning "Redis failed to start"
    fi
    
    if docker ps | grep -q postgres; then
        print_success "PostgreSQL is running"
    else
        print_warning "PostgreSQL failed to start"
    fi
fi

# Run health check
read -p "Do you want to run system health check? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running health check..."
    $PYTHON_CMD scripts/health_check.py
fi

# Create run scripts
print_status "Creating run scripts..."

# Create run_arbitrage.sh
cat > run_arbitrage.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python main.py --mode arbitrage --symbols BTC/USDT ETH/USDT BNB/USDT
EOF
chmod +x run_arbitrage.sh

# Create run_monitor.sh
cat > run_monitor.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python main.py --mode monitor --symbols BTC/USDT ETH/USDT BNB/USDT
EOF
chmod +x run_monitor.sh

# Create run_api.sh
cat > run_api.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
EOF
chmod +x run_api.sh

# Create run_performance_monitor.sh
cat > run_performance_monitor.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python scripts/monitor_performance.py
EOF
chmod +x run_performance_monitor.sh

print_success "Run scripts created"

# Final summary
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}       SETUP COMPLETED SUCCESSFULLY! 🎉              ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo
echo "Next steps:"
echo "1. Edit .env file and add your exchange API keys"
echo "2. Start the API server: ./run_api.sh"
echo "3. Run health check: python scripts/health_check.py"
echo "4. Start arbitrage bot: ./run_arbitrage.sh"
echo "5. Monitor performance: ./run_performance_monitor.sh"
echo
echo "For monitoring only (no trading): ./run_monitor.sh"
echo
echo "Documentation: See README.md for detailed instructions"
echo
print_warning "IMPORTANT: Never commit your .env file with real API keys!"
echo