from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
# data = load_iris(as_frame=True)
# df = data.frame
# df['Target'] = df["target"]
# df = df.drop(columns=["target"])
# train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Target"])
# train.to_csv("Train.csv", index=False)
# test.to_csv("Test.csv", index=False)

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
df['Target'] = df["target"]
df = df.drop(columns=["target"])
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Target"])
train.to_csv("Train.csv", index=False)
test.to_csv("Test.csv", index=False)