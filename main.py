from get_km_price_data import get_km_price_data
from train import train
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from estimate_price import estimate


def main():
    filename = 'data.csv'
    data = get_km_price_data(filename)
    model = train(np.array(data.loc[:, 'km']), np.array(data.loc[:, 'price']))
    thetas = model.gradient_descent(200, 0.0000000001)
    print(thetas)
    plt.figure()
    sns.scatterplot(x=data.loc[:, 'km'], y=data.loc[:, 'price'])
    plt.show()
    
    estimate_model = estimate(thetas[0], thetas[1])
    km = np.array([float(x) for x in range(0, 250000, 100)])
    price = np.array([estimate_model.estimate_price(x) for x in km])
    plt.figure()
    sns.scatterplot(x=km, y=price)
    plt.show()
    
    plt.figure()
    sns.lineplot(model.history)
    plt.show()


if __name__ == "__main__":
    main()