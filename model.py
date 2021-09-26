from scipy.sparse import data
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau
import data_utils
from sklearn import preprocessing

# Get training data
X = data_utils.get_training_data()
# X = X[:,1:3]
# X = preprocessing.normalize(X)
print(X.shape)
x_train, y_train, x_test, y_test = data_utils.split_data(X)
x_train = x_train[:,1:3]
x_test = x_test[:,1:3]

# Create model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

# Train model

model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data = (x_test,y_test), epochs=900, batch_size=512)

model.save("model.hdf5")