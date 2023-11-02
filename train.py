from estimate_price import estimate
import numpy as np


class train:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.theta0 = 7000
        self.theta1 = 0.02
        self.history = np.array([])
        self.m = float(len(x))
    
    def gradient_descent(self, epoch, lr):
        self.history = np.zeros(epoch)
        for i in range(epoch):
            tmp_theta0, tmp_theta1 = 0.0, 0.0
            estimate_model = estimate(self.theta0, self.theta1)
            each_diff = estimate_model.estimate_price(self.x) - self.y
            tmp_theta0 = lr / self.m * np.sum(each_diff)
            tmp_theta1 = lr / self.m * np.sum(each_diff * self.x)
            self.theta0 = self.theta0 - tmp_theta0
            self.theta1 = self.theta1 - tmp_theta1
            print(f"{i} epoch loss: {self.loss()}")
            self.history[i] = self.loss()
        return self.theta0, self.theta1
    
    def loss(self):
        estimate_model = estimate(self.theta0, self.theta1)
        each_diff = (estimate_model.estimate_price(self.x) - self.y) ** 2
        total_loss = sum(each_diff) / self.m
        return total_loss
        
