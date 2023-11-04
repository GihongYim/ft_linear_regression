import pandas as pd
import numpy as np
from predict import predict_price
from linear_regression import Linear_regression

def main():
    data = pd.read_csv('data.csv')
    km =  data.loc[:,'km'].astype('float')
    price = data.loc[:, 'price'].astype('float')
    model = Linear_regression(x=km, y=price)
    model.train(epochs=20000, lr=0.0001)
    theta_history = model.theta_history
    theta0, theta1 = model.theta0, model.theta1
    data_x = np.linspace(np.min(km), np.max(km), num=1000)
    data_y = []
    

if __name__ == "__main__":
    main()