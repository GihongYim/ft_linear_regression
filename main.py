import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from predict import predict_price
from linear_regression import Linear_regression

def main():
    data = pd.read_csv('data.csv')
    km =  data.loc[:,'km'].astype('float')
    price = data.loc[:, 'price'].astype('float')
    model = Linear_regression(x=km, y=price)
    model.train(epochs=10000, lr=0.01)
    # theta_history = model.theta_history
    # theta0, theta1 = model.theta0, model.theta1
    fig = plt.figure()
    sns.scatterplot(x=km, y=price)
    
    for theta0, theta1 in model.theta_history:
        model.theta0 = theta0
        model.theta1 = theta1
        plt.xlim(model.x_min, model.x_max)
        plt.ylim(model.y_min, model.y_max)
        data_x = np.linspace(np.min(km), np.max(km), num=1000)
        data_y = [model.predict(x) for x in data_x]
        fig.clear()
        sns.lineplot(x=data_x, y=data_y)
        plt.pause(0.01)
    plt.show()    
    plt.figure()
    sns.lineplot(model.loss_history)
    plt.show()
    
    

if __name__ == "__main__":
    main()