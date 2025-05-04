#!/bin/bash

# Enable debugging
set -x

# Set up Python environment
export PYTHONPATH=/opt/render/project/src
export PATH=/opt/render/project/src/.venv/bin:$PATH

# Print environment information
echo "Current PATH: $PATH"
echo "Current PYTHONPATH: $PYTHONPATH"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Gunicorn version: $(gunicorn --version)"

# Activate virtual environment
source /opt/render/project/src/.venv/bin/activate

# Install dependencies if needed
pip install -r requirements.txt

# Start the application
echo "Starting Gunicorn application..."
exec gunicorn app:app 