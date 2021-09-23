from keras.models import load_model
import pandas as pd
import data_utils
import numpy as np

DATA_PATH = "training.csv"
raw = pd.read_csv('data/Interview_data.csv')

model = load_model('model')
# missing_prices = data_utils.engineer_features(data_utils.get_missing_payments(raw))
# arr = missing_prices.to_numpy().astype('float32')
# print(arr.shape)
# for i in range(len(missing_prices)):
#     pred = model.predict(arr[i])
#     print("Prediction: ", pred)

x = np.array([[39,251.083333333333,24,8,7,1998,4]])
x2 = np.array([[61,244.9575,12,14,8,1998,7]])
print(x.shape)
pred = model.predict(x)
print(pred)
print(model.predict(x2))