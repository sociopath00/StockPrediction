DATA_COLUMNS = ["dates", "open", "high", "low", "close", "wap", "no_shares",
                "no_trades", "no_turnover", "qty", "perc_del_trade", "diff_high_low", "diff_open_close"]

DATA = "data/RIL_equity_3_years.csv"

TARGET = "high"

NO_OF_FEATURES = 75

MODEL_PATH = "models/lstm_model.h5"

# model hyper-parameters
EPOCHS = 200
BATCH_SIZE = 32
LOSS_FUNCTION = "mean_squared_error"
OPTIMIZER = "adam"

TEST_DATA = "data/test.csv"
PLOT = "output/graph.png"

NO_OF_PREDICTIONS = 3
