#!/bin/bash
set -euo pipefail

find . -name "*.py" -print0 | xargs -0 yapf -i
echo "All Python code formatted."
