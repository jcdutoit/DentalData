from sklearn.svm import OneClassSVM
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import data_utils

RAW_DATA = "data/Interview_data.csv"

data = data_utils.get_temporal_data(pd.read_csv(RAW_DATA)).T
print(data.shape)
plt.scatter(data[:,0], data[:,1])
plt.show()

svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
svm.fit(data)
pred = svm.predict(data)
scores = svm.score_samples(data)

thresh = np.quantile(scores, 0.03)

outliers = np.where(scores <= thresh)
out_vals = data[outliers]

plt.scatter(data[:,0], data[:,1])
plt.scatter(out_vals[:,0], out_vals[:,1], color = 'r')
plt.show()