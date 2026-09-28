#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
"${PYTHON:-python3}" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m gradforge.experiment
