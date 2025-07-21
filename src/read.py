import csv
import os
import joblib
import json


def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None


def load_params(params_path):
    if os.path.exists(params_path):
        with open(params_path, 'r') as file:
            return json.load(file)
    else:
        return None
