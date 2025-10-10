#!/bin/bash

# CIRR Multi-Image Understanding Test - Background Runner
# This script runs the evaluation in the background with logging

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/eval_background_${TIMESTAMP}.log"
ERROR_LOG="logs/eval_error_${TIMESTAMP}.log"
PID_FILE="logs/eval_background.pid"

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi
}

# Set trap to cleanup on script exit
trap cleanup EXIT

echo "=========================================="
echo "CIRR Multi-Image Understanding Test"
echo "Background Execution Started"
echo "=========================================="
echo "Start time: $(date)"
echo "Script directory: $SCRIPT_DIR"
echo "Log file: $LOG_FILE"
echo "Error log: $ERROR_LOG"
echo "PID file: $PID_FILE"
echo ""
echo "Parameters:"
echo "  --num_groups: 50"
echo "  --max_pairs_per_group: 6"
echo "=========================================="

# Start the evaluation in background
nohup python eval_found.py \
    --num_groups 50 \
    --max_pairs_per_group 6 \
    > "$LOG_FILE" 2> "$ERROR_LOG" &

# Get the process ID
EVAL_PID=$!

# Save PID to file
echo $EVAL_PID > "$PID_FILE"

echo "Evaluation started in background with PID: $EVAL_PID"
echo "PID saved to: $PID_FILE"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check for errors:"
echo "  tail -f $ERROR_LOG"
echo ""
echo "To stop the evaluation:"
echo "  kill $EVAL_PID"
echo "  # or"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "To check if still running:"
echo "  ps -p $EVAL_PID"
echo "=========================================="

# Wait a moment to check if the process started successfully
sleep 3

if ps -p $EVAL_PID > /dev/null; then
    echo "✓ Process is running successfully"
    echo "Background execution initiated. Check logs for progress."
else
    echo "✗ Process failed to start or exited immediately"
    echo "Check error log: $ERROR_LOG"
    exit 1
fi 