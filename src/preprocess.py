from config import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from sklearn.metrics import mean_squared_error


class LSTMPreprocessor:
    def __init__(self, data: pd.DataFrame, target: str):
        """
        :param data: input data as pandas DataFrame
        :param target: name of the column to be treated as target/output variable
        """
        self.target = target

        # subset our data with index and target column
        # i.e convert it into univariate data
        self.data = data.reset_index()[self.target]
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def split_dataset(self, train_size=0.8,):
        """
        :param train_size: size of the train data ( test data size will be 1-train_size)
        :return: train_data, test_data
        """
        if 0 > train_size > 1:
            raise ValueError("Train size must be between 0 to 1")

        data = self.scaler.fit_transform(np.array(self.data).reshape(-1, 1))

        train_length = int(len(data)*train_size)

        train_data = data[:train_length, :]
        test_data = data[train_length:, :]

        return train_data, test_data

    @staticmethod
    def create_features(data: pd.DataFrame, time_step: int = 1):
        """
        :param data: data as pandas DataFrame
        :param time_step: no of rows to be used for
        :return:
        """
        x_data, y_data = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            x_data.append(a)
            y_data.append(data[i + time_step, 0])

        # convert data into numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        print(x_data.shape)

        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
        return x_data, y_data

    def inverse_normalize(self, data):
        """
        :param data: scaled data as numpy array or dataframe column
        :return:
        """
        data = self.scaler.inverse_transform(data)
        return data

    def predict_values(self, data, model, inverse=False):
        """
        :param data: test or validation data
        :param model: trained model
        :param inverse: if data is scaled, set it as True
        :return: predictions
        """
        predictions = model.predict(data)
        if inverse:
            predictions = self.inverse_normalize(predictions)

        return predictions

    @staticmethod
    def model_performance(y_true, y_pred, method="mse"):
        """
        :param y_true: actual values
        :param y_pred: predicted values
        :param method: rmse or mse
        :return: result
        """
        result = mean_squared_error(y_true, y_pred)
        if method == "rmse":
            result = math.sqrt(result)
        return result


if __name__ == "__main__":
    pass



