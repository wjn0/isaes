#!/bin/bash

sbatch --time=24:00:00 -c 8 --mem=64G --job-name=mlflow --time=24:00:00 --wrap="bash -c 'uv run mlflow ui --host 0.0.0.0 --port 5000'"
echo 'ssh -N -L 5000:$(squeue --user=$USER --name=mlflow -o "%N" -h):5000 $(hostname)'
