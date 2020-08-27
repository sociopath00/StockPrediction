import pandas as pd
from config import *
from .preprocess import LSTMPreprocessor
from .ml_models import LSTMStackN
import math
from sklearn.metrics import mean_squared_error


df = pd.read_csv(DATA)
df.columns = DATA_COLUMNS


pp = LSTMPreprocessor(df, "close")

train, test = pp.split_dataset()

x_train, y_train = pp.create_features(train, NO_OF_FEATURES)
x_test, y_test = pp.create_features(test, NO_OF_FEATURES)

print(x_train)

model = LSTMStackN.build()
model.compile(loss='mean_squared_error', optimizer='adam',)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=64, verbose=1)

train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

train_predict = pp.inverse_normalize(train_predict)
test_predict = pp.inverse_normalize(test_predict)

## Calculate RMSE performance metric
rmse = math.sqrt(mean_squared_error(y_train, train_predict))
print(rmse)


