"""
Tests Module - Unit, integration and performance tests
"""

import os
import sys

# Add project root to path for test imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)