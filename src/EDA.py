import pandas as pd
import numpy as np


data = pd.read_csv("../data/[new] yancheng_train_20171226.csv")
print(data.columns)

print(data[data["class_id"] ==  103507 & data["sale_date"] == 201609])