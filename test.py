import pandas as pd

train = pd.read_csv("data/input/train_raw.csv")
val = pd.read_csv("data/input/dev_raw.csv")
test = pd.read_csv("data/input/test_raw.csv")

print(len(train))
print(len(val))
print(len(test))