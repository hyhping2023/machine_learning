import pandas as pd
import torch

data = pd.read_csv('../data/house_tiny.csv')
print(data)

inputs, outputs = data.iloc[:,0:2], data.iloc[:,2]
inputs = inputs.fillna(inputs.select_dtypes(include=[int, float]).mean())
print(inputs)

inputs = data.iloc[:,0:1]
inputs = inputs.fillna(inputs.mean())
inputs, outputs = torch.tensor(inputs.to_numpy(dtype=int)), torch.tensor(outputs.to_numpy(dtype=int))
outputs = outputs.reshape(4,1)
results = torch.cat((inputs, outputs), dim = 1)

print(results)
# print(inputs)
# print(outputs)

def drop_na_most(data):
    nan = data.isna().sum()
    nan = nan.to_dict()
    max_key = max(nan, key=nan.get)
    return data.drop(columns=max_key)

data = drop_na_most(data)
print(data)

tensor1 = torch.tensor(data.values, dtype=int)
tensor2 = torch.tensor(data.to_numpy(), dtype=int)
print(tensor1)
print(tensor2)