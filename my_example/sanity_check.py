import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('logP_dataset.csv')

fingerprints = []
for mol_id in df['mol_id'].values:
	with open(os.path.join('embeddings', f'{mol_id}_embeddings.pkl'), 'rb') as handler:
		d = pickle.load(handler)
		fingerprints.append(np.mean(d['atoms'], axis=0))
fingerprints = np.vstack(fingerprints)
logp = df['logp'].values

test_x, train_x = fingerprints[:100], fingerprints[100:]
test_y, train_y = logp[:100], logp[100:]

model = LinearRegression()
model.fit(train_x, train_y)

test_preds = model.predict(test_x)
plt.scatter(test_y, test_preds)
plt.xlabel('Target logP')
plt.ylabel('Predicted logP')
plt.show()

