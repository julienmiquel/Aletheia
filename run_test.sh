#!/bin/bash

# Aletheia Test Runner

echo "========================================"
echo "    Aletheia AI Detector - Test Suite   "
echo "========================================"

# Check if pytest is installed
if ! ./venv/bin/python -c "import pytest" &> /dev/null; then
    echo "Error: pytest is not installed. Run 'pip install pytest'."
    exit 1
fi

echo "Running tests with pytest..."
./venv/bin/python -m pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed."
    exit 1
fi
