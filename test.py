import pandas as pd
import numpy as np


df = pd.read_csv("data/input/dev_raw.csv")
print(df["Toxicity"].value_counts())