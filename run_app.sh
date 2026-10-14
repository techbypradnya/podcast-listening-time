#!/usr/bin/env bash
# Launch the Dropoff Studio UI.  Usage: ./run_app.sh
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PYTHONPATH"
[ -f models/podcast_model.keras ] || { echo "No trained model found - training one now (about 2 min)..."; python -m podcast_ml.train --epochs 60; }
streamlit run app/app.py
