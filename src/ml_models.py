from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


class LSTMStackN:
    @staticmethod
    def build(input_shape=(30, 1), n=1):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))

        # add layers based on value of n
        for i in range(n):
            model.add(LSTM(50, return_sequences=True))

        model.add(LSTM(50))
        model.add(Dense(1))

        return model


class SimpleLSTM:
    @staticmethod
    def build(input_shape=(50, 1)):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(100))
        model.add(Dense(1))

        return model


