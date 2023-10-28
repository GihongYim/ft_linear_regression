import numpy as np


def train(data, num_of_epoch, learning_rate):
    theta0, theta1 = 0, 0
    for _ in range(num_of_epoch):
        tmp_theta0, tmp_theta1 = 0.0, 0.0
        for row_index in range(0, len(data)):
            row = data.loc[row_index]
            loss = estimate_price(theta0, theta1, row['km']) - row['price']
            tmp_theta0 += loss
            tmp_theta1 += loss * row['km']
        theta0 = theta0 - learning_rate / len(data) * tmp_theta0
        theta1 = theta1 - learning_rate / len(data) * tmp_theta1
        print(theta0)
        print(theta1)
    return theta0, theta1


def estimate_price(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)
