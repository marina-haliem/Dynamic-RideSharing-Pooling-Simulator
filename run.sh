#!/bin/bash

# Installation
pip install -r requirements.txt

# Runing the simulation
python ./simulator_driver.py

# Parse the results
python ./parse_results.py
