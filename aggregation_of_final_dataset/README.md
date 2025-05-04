# How to build the final dataset
1. Create a virtual environment with `requirements.txt`
2. Change `raw_dir`, `processed_dir`, `intermediate_dir` in `aggregation_of_final_dataset/settings.py`
3. Run `export PYTHONPATH="$(pwd):PYTHONPATH"`
4. Aggregate each dataset by following the guides below

### Project Natick
1. Simply run `python aggregation_of_final_dataset/project_natick.py`

### Roboflow Fish
1. Simply run `python aggregation_of_final_dataset/roboflow_fish.py`

### Zebrafish
1. Simply run `python aggregation_of_final_dataset/zebrafish.py`