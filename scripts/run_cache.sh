#!/bin/bash

# Start tmux session and run cache_activations.py
tmux new-session -d -s cache_session "cd /home/can/dynamic_representations && python cache/cache_activations.py"

echo "Started cache_activations.py in tmux session 'cache_session'"
echo "Attach with: tmux attach -t cache_session"