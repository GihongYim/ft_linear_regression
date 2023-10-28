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
        tmp_theta0 /= len(data)
        tmp_theta1 /= len(data)
        print(tmp_theta0)
        print(tmp_theta1)
        break
        tmp_theta0 *= learning_rate
        tmp_theta1 *= learning_rate
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
    return theta0, theta1


def estimate_price(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)
