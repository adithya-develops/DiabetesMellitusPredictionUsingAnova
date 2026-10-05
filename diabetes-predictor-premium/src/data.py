from pathlib import Path
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df["hypertension"] = df["hypertension"].astype(int)
    df["heart_disease"] = df["heart_disease"].astype(int)
    df["diabetes"] = df["diabetes"].astype(int)
    return df
