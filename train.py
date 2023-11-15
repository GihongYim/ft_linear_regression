#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linear_regression import Linear_regression

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("input your filename")
        exit(0)
    try:
        data = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(f"{e}")
        exit(1)
    try:
        km =  data.loc[:,'km'].astype('float')
        price = data.loc[:, 'price'].astype('float')
    except Exception as e:
        print(f"{e} column not found. check your csv file")
        exit(0)
    model = Linear_regression(x=km, y=price)
    model.train(epochs=2000, lr=0.03)
    fig = plt.figure()
    
    epoch = 1
    for theta0, theta1 in model.theta_history:
        model.theta0 = theta0
        model.theta1 = theta1
        if epoch % 20 == 0 or epoch == len(model.theta_history):
            plt.xlim(model.x_min, model.x_max)
            plt.ylim(model.y_min, model.y_max)
            data_x = np.linspace(np.min(km), np.max(km), num=1000)
            data_y = [model.predict(x) for x in data_x]
            fig.clear()
            sns.scatterplot(x=km, y=price, color='red')
            sns.lineplot(x=data_x, y=data_y, color='blue').set(title=f"{epoch} epoch")
            plt.legend(["train data", "predict"])
            plt.pause(0.0001)
        epoch += 1
    plt.show()    
    plt.figure()
    sns.lineplot(model.loss_history).set(title="normalized data loss")
    plt.legend(['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    error_dist = [y - model.predict(x) for x, y in zip(km, price)]
    sns.histplot(data=error_dist).set(title="error distribution")
    plt.xlabel('y - f(x)')
    plt.ylabel('count')
    plt.show()