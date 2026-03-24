#!/bin/bash
# Stop the web frontend service

echo "Stopping JittorGeometric web frontend..."

# Find and kill the web frontend process
pkill -f "python web_frontend.py"

# If pkill fails, use ps and kill
if [ $? -ne 0 ]; then
    echo "pkill failed, trying ps..."
    PID=$(ps aux | grep "python web_frontend.py" | grep -v grep | awk '{print $2}')
    if [ -n "$PID" ]; then
        kill $PID
        echo "Killed process $PID"
    else
        echo "No web frontend process found"
    fi
else
    echo "Web frontend stopped successfully"
fi
