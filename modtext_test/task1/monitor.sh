#!/bin/bash

# CIRR Evaluation Monitor Script
# This script helps monitor the background evaluation process

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="logs/eval_background.pid"

echo "=========================================="
echo "CIRR Evaluation Monitor"
echo "=========================================="

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "❌ No background process found (PID file missing)"
    echo "Start evaluation with: ./run_background.sh"
    exit 1
fi

# Read PID
EVAL_PID=$(cat "$PID_FILE")
echo "📋 Process ID: $EVAL_PID"

# Check if process is running
if ps -p $EVAL_PID > /dev/null; then
    echo "✅ Status: RUNNING"
    
    # Show process info
    echo ""
    echo "📊 Process Information:"
    ps -p $EVAL_PID -o pid,ppid,cmd,etime,pcpu,pmem
    
    echo ""
    echo "📁 Log Files:"
    
    # Find latest log files
    LATEST_LOG=$(ls -t logs/eval_background_*.log 2>/dev/null | head -1)
    LATEST_ERROR=$(ls -t logs/eval_error_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LOG" ]; then
        echo "  📝 Output log: $LATEST_LOG"
        echo "  📏 Log size: $(du -h "$LATEST_LOG" | cut -f1)"
        echo "  📄 Last 3 lines:"
        tail -3 "$LATEST_LOG" | sed 's/^/    /'
    fi
    
    if [ -n "$LATEST_ERROR" ] && [ -s "$LATEST_ERROR" ]; then
        echo ""
        echo "  ⚠️  Error log: $LATEST_ERROR"
        echo "  📏 Error log size: $(du -h "$LATEST_ERROR" | cut -f1)"
        echo "  📄 Last 3 error lines:"
        tail -3 "$LATEST_ERROR" | sed 's/^/    /'
    fi
    
    echo ""
    echo "🔍 Monitoring Commands:"
    echo "  Monitor output: tail -f $LATEST_LOG"
    echo "  Monitor errors: tail -f $LATEST_ERROR"
    echo "  Stop process:   kill $EVAL_PID"
    
else
    echo "❌ Status: NOT RUNNING"
    echo "Process with PID $EVAL_PID is not running"
    
    # Check if there are recent log files
    LATEST_LOG=$(ls -t logs/eval_background_*.log 2>/dev/null | head -1)
    LATEST_ERROR=$(ls -t logs/eval_error_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_ERROR" ] && [ -s "$LATEST_ERROR" ]; then
        echo ""
        echo "⚠️  Check error log for issues:"
        echo "  $LATEST_ERROR"
        echo ""
        echo "Last 5 error lines:"
        tail -5 "$LATEST_ERROR" | sed 's/^/  /'
    fi
    
    # Clean up PID file
    rm -f "$PID_FILE"
    echo ""
    echo "🔄 To restart: ./run_background.sh"
fi

echo "==========================================" 