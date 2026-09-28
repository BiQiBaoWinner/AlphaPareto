#!/bin/bash
# OpenCode Agent wrapper for AutoRL-Bench

echo "=== OpenCode Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"
echo "Grading Server: $GRADING_SERVER_URL"
echo "Output Dir: $OUTPUT_DIR"

# Load .env when running from RD-Agent root
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

# opencode-rl checkout; override with OPENCODE_RL_ROOT
OPENCODE_RL_ROOT="${OPENCODE_RL_ROOT:-/Your/Path/To/opencode-rl}"

export OPENCODE_MODEL="${OPENCODE_MODEL:-${CHAT_MODEL:-gpt-5}}"
echo "OpenCode Model: $OPENCODE_MODEL"

export PYTHONUNBUFFERED=1

export PATH="$HOME/.opencode/bin:$PATH"

if [ -n "$TRAINING_PYTHON" ]; then
    TRAINING_BIN_DIR="$(dirname "$TRAINING_PYTHON")"
    export PATH="$TRAINING_BIN_DIR:$PATH"
    echo "Training env bin: $TRAINING_BIN_DIR (prepended to PATH)"
fi

PYTHON="${OPENCODE_PYTHON:-python3}"
echo "Python: $PYTHON"

export XDG_CONFIG_HOME="${OPENCODE_RL_ROOT}/.opencode-config"
mkdir -p "$XDG_CONFIG_HOME/opencode"
cat > "$XDG_CONFIG_HOME/opencode/opencode.json" <<EOCFG
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "openai": {
      "npm": "@ai-sdk/openai",
      "name": "Auto-configured",
      "options": {
        "baseURL": "${OPENAI_API_BASE}",
        "apiKey": "${OPENAI_API_KEY}"
      },
      "models": {
        "${OPENCODE_MODEL}": { "name": "${OPENCODE_MODEL}" }
      }
    }
  }
}
EOCFG

cd "$OPENCODE_RL_ROOT"

# Use exec to REPLACE bash with python3, so signals go directly to python3
# without an intermediate bash process. This avoids double signal delivery.
exec "$PYTHON" main.py \
    --benchmark "$TASK" \
    --base-model "$BASE_MODEL" \
    --run-dir "$WORKSPACE" \
    --max-iterations ${MAX_ITERATIONS:-5} \
    --max-retries ${MAX_RETRIES:-20} \
    --training-timeout ${TRAINING_TIMEOUT:-7200} \
    --stale-timeout ${STALE_TIMEOUT:-1800} \
    --http-timeout ${HTTP_TIMEOUT:-600} \
    --eval-timeout ${EVAL_TIMEOUT:-7200} \
    --max-agent-steps ${MAX_AGENT_STEPS:-25}
