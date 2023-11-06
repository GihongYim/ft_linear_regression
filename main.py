import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from predict import predict_price
from linear_regression import Linear_regression

def main():
    try:
        data = pd.read_csv('data.csv')
    except Exception as e:
        print(f"{e}")
        return
    km =  data.loc[:,'km'].astype('float')
    price = data.loc[:, 'price'].astype('float')
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
            sns.scatterplot(x=km, y=price)
            sns.lineplot(x=data_x, y=data_y).set(title=f"{epoch} epoch")
            plt.pause(0.0001)
        epoch += 1
    plt.show()    
    plt.figure()
    sns.lineplot(model.loss_history).set(title="loss")
    plt.show()
    error_dist = [y - model.predict(x) for x, y in zip(km, price)]
    sns.histplot(data=error_dist).set(title="error : y - f(x)")
    plt.show()
    
    

if __name__ == "__main__":
    main()