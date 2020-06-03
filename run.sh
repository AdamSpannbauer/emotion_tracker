#!/bin/bash

source venv/bin/activate
python emotion_tracker.py
python combine_data.py
deactivate
