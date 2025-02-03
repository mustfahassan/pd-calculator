#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start Gunicorn
exec gunicorn -c gunicorn_config.py run:app