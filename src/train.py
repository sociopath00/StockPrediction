import argparse
import pandas as pd
from .ml_models import get_model
from .preprocess import LSTMPreprocessor
from config import *
import tensorflow as tf
import os
import matplotlib.pylab as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", default=DATA, help="path to the data")
ap.add_argument("-t", "--target", default=TARGET, help="column on which predictions to be made")
ap.add_argument("-p", "--model", default=MODEL, help="model to be used")
ap.add_argument("-g", "--graph", default=False, help="Plot the graph or not")
args = vars(ap.parse_args())


# read the data
df = pd.read_csv(args["data"])

# replace the column names
df.columns = DATA_COLUMNS

df["dates"] = pd.to_datetime(df["dates"])
df = df.sort_values(by="dates").reset_index(drop=True)

# print(df.head())
MODEL_PATH = args["model"] + ".h5"
MODEL_PATH = os.path.join("models", MODEL_PATH)

# preprocessing on a model
pp = LSTMPreprocessor(df, args["target"])

train, test = pp.split_dataset()

# we are storing test data in csv for further prediction
pd.DataFrame(pp.inverse_normalize(test)).to_csv(TEST_DATA, index=False)

x_train, y_train = pp.create_features(train, NO_OF_FEATURES)
x_test, y_test = pp.create_features(test, NO_OF_FEATURES)

# build compile and train the model
# model = LSTMStackN.build()
# model.compile(loss='mean_squared_error', optimizer='adam',)
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# build and compile the model
model = get_model(args["model"])
model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# save the model
model.save(MODEL_PATH)

# print the rmse score
prediction = pp.predict_values(x_test, model, inverse=True)
rmse = pp.model_performance(y_test, prediction, method="rmse")

print("[INFO] RMSE on validation data: ", rmse)

if args["graph"] == "True":
    plt.plot(pp.inverse_normalize(y_test.reshape(y_test.shape[0], -1)))
    plt.plot(prediction)
    plt.title("Actual vs Predicted values")
    plt.show()
    plt.savefig(PLOT)
