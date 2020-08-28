# StockPrediction
  This module aims to predict stock prices based on past data

### Prerequisite:
- Python3
- Git


## Steps

- Download git repo on your machine

    `$ git clone https://github.com/sociopath00/StockPrediction.git`


- Change working directory

    `$ cd StockPrediction`
    
    
- Install Requirements and Dependancies

    `$ pip install -r requirements.txt`
`


- Train the model

    `$ python -m src.train`
    
    *Note*: If you wish to train with different dataset mention the path with `-d` argument, target variable with `-t` and model path with `-p`
    
    eg.  `$ python -m src.train -d data/train.csv -p stacked_lstm -t close`
    

- Predict the values on Test data

     `$ python -m src.predict -p stacked_lstm`
     
     
    
    
 
