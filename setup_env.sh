#!/usr/bin/env bash
# Recreates the "multi_agent_mpc" conda environment this project uses
# (Python 3.11, dependencies from requirements.txt).
set -euo pipefail

ENV_NAME="multi_agent_mpc"
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
conda run -n "$ENV_NAME" pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Environment '$ENV_NAME' created. Activate it with: conda activate $ENV_NAME"
