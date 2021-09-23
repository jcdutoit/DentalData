from scipy.sparse import data
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import data_utils
from sklearn.preprocessing import StandardScaler

DATA_PATH = "training.csv"

raw = pd.read_csv(DATA_PATH)
X = data_utils.engineer_features(raw)
print(X.loc[0])
sc = StandardScaler()
X = sc.fit_transform(X)

x_train, y_train, x_test, y_test = data_utils.split_data(X)
print(x_train[0])
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = Sequential()
model.add(Dense(32, input_dim=7, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

# print(x_train[0])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
model.compile(loss=keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data = (x_test,y_test), epochs=800, batch_size=512)

model.save('model')