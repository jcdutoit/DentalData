import os
import numpy as np
from pandas.core.indexes.base import ensure_index
import keras
import data_utils

if not os.path.exists('model.hdf5'):
    print("Model not found. Please run model.py to train the model")
    quit()

model = keras.models.load_model('model.hdf5')

missing = data_utils.get_missing()
missing = missing[:,1:3]
for x in missing:
    print(model.predict(np.array([x])))

