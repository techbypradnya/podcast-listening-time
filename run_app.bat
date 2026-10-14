@echo off
REM Launch the Dropoff Studio UI (Windows)
cd /d "%~dp0"
set PYTHONPATH=%CD%\src;%PYTHONPATH%
if not exist models\podcast_model.keras (
  echo No trained model found - training one now ^(about 2 min^)...
  python -m podcast_ml.train --epochs 60
)
streamlit run app\app.py
