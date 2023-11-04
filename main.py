from get_km_price_data import get_km_price_data
from linear_regression import Linear_regression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from predict_price import predict


def main():
    filename = 'data.csv'
    data = get_km_price_data(filename)
    km = np.array(data.loc[:, 'km'])
    price = np.array(data.loc[:, 'price'])
    model = Linear_regression(km, price, predict)
    thetas = model.gradient_descent(50000, 0.001)
    print(thetas)
    fig, ax = plt.subplots(nrows=2)
    sns.scatterplot(x=data.loc[:, 'km'], y=data.loc[:, 'price'], ax=ax[0])    
    km = np.array([float(x) for x in range(0, 250000, 100)])
    price = np.array([model.predict(x) for x in km])
    sns.lineplot(x=km, y=price, ax=ax[0])        
    sns.lineplot(model.history, ax=ax[1])
    plt.show()


if __name__ == "__main__":
    main()