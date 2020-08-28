import os
import argparse
import pandas as pd
import numpy as np
from config import *
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", default=TEST_DATA, help="path to the dataset")
ap.add_argument("-p", "--model", default=MODEL, help="model to be used")
args = vars(ap.parse_args())

# read data
data = pd.read_csv(args["data"])

start = len(data)-NO_OF_FEATURES
data = data[start:]

MODEL_PATH = os.path.join("models", args["model"]+".h5")

# load model
model = load_model(MODEL_PATH)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

final_predictions = []

for i in range(NO_OF_PREDICTIONS):
    # take only last N(no of features) values
    scaled_data = scaled_data[i:]
    x = scaled_data.T
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y_pred = model.predict(x)
    scaled_data = np.append(scaled_data, y_pred, axis=0)

    # print(scaled_data)
    y_pred = scaler.inverse_transform(y_pred)
    final_predictions.extend(y_pred[0])


print(final_predictions)
