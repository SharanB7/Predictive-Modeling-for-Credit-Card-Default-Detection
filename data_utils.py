# data_utils.py
import pandas as pd

def load_and_preprocess_data(file_path):
    # Importing the Credit Card Default Taiwan Dataset
    dataset = pd.read_csv(file_path)
    dataset = dataset.sample(n=30000, replace=False, random_state=1)

    return dataset
