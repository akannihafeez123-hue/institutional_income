#!/usr/bin/env bash
set -e

echo "=== ENVIRONMENT ==="
python --version
pip --version

echo "=== INSTALL DEPENDENCIES ==="
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "=== PY COMPILE (syntax check) ==="
python -m py_compile legendary_empire_app.py || { echo "py_compile failed"; exit 1; }

echo "=== IMPORT TEST (prints traceback on failure) ==="
python legendary_empire_app.py debug-imports || { echo "debug-imports failed"; exit 1; }

echo "=== CONFIG TEST ==="
python legendary_empire_app.py test-config || { echo "config test failed"; exit 1; }

echo "=== BUILD CHECK PASSED ==="
