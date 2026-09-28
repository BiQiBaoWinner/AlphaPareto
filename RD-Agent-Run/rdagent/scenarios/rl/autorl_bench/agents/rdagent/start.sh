#!/bin/bash
# RD-Agent wrapper for AutoRL-Bench

echo "=== RD-Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

export RL_FILE_PATH=$(dirname $(dirname $MODEL_PATH))
echo "RL_FILE_PATH: $RL_FILE_PATH"

python -m rdagent.app.rl.loop \
    --base-model "$BASE_MODEL" \
    --benchmark "$TASK" \
    --step-n $STEP_N \
    --loop-n $LOOP_N
