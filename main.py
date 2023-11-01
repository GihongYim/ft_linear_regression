import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from graph import show_data_graph
from train_model import train, estimate_price


def main():
    """
        main function for ft_linear_regression
    """
    data = pd.read_csv("data.csv")
    show_data_graph(data)
    theta0, theta1 = train(data, 5, 0.0005)
    x_vec = [x * 1000 for x in np.arange(0, 300000, 10000)]
    y_vec = [estimate_price(theta0, theta1, x) * 1000 for x in x_vec]
    plt.figure()
    sns.lineplot(x=x_vec, y=y_vec)
    plt.show()


if __name__ == "__main__":
    main()