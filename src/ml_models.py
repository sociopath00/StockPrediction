from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, LSTM, Dense
from config import *


class StackLSTM:
    @staticmethod
    def build(input_shape=(NO_OF_FEATURES, 1), n=1, dropout=False):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))

        # add layers based on value of n
        for i in range(n):
            model.add(LSTM(50, return_sequences=True))
            if dropout:
                model.add(Dropout(0.2))

        model.add(LSTM(50))
        model.add(Dense(1))

        return model


class SimpleLSTM:
    @staticmethod
    def build(input_shape=(NO_OF_FEATURES, 1)):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(100))
        model.add(Dense(1))

        return model


class SimpleGRU:
    @staticmethod
    def build(input_shape=(NO_OF_FEATURES, 1)):
        # gru = GRU(10)

        model = Sequential()
        model.add(GRU(25, return_sequences=True, input_shape=input_shape))
        model.add(GRU(25))
        model.add(Dense(1))

        return model


class StackGRU:
    @staticmethod
    def build(input_shape=(NO_OF_FEATURES, 1), n=1, dropout=False):
        # gru = GRU(10)

        model = Sequential()
        model.add(GRU(25, return_sequences=True, input_shape=input_shape))
        for i in range(n):
            model.add(GRU(25, return_sequences=True))
            if dropout:
                model.add(Dropout(0.2))
        model.add(GRU(25))
        model.add(Dense(1))

        return model


def get_model(model_name, no_features=NO_OF_FEATURES, **kwargs):
    m = {"lstm": SimpleLSTM, "stacked_lstm": StackLSTM,
         "gru": SimpleGRU, "stacked_gru": StackGRU}

    if model_name == "gru":
        M = m[model_name]
        model = M.build(input_shape=(no_features, 1), **kwargs)
        return model

    try:
        M = m[model_name]
    except KeyError:
        raise ValueError("Select the model name correctly")

    model = M.build(input_shape=(no_features, 1), **kwargs)

    return model
