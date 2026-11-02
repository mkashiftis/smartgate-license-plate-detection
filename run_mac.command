#!/bin/bash
# SmartGate - double-click to run on a Mac (or: bash run_mac.command)
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "First run: installing everything. This takes 5-10 minutes and needs internet."
  python3 -m venv .venv || { echo "Python 3 is not installed. Get it from https://www.python.org/downloads/"; read -p "Press Enter to close"; exit 1; }
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt || { echo "Install failed - see the message above."; read -p "Press Enter to close"; exit 1; }
  echo "Downloading the text-reading model (about 100 MB, one time only)..."
  ./.venv/bin/python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"
fi

( sleep 5; open "http://localhost:8000" ) &
./.venv/bin/python app.py "$@"
