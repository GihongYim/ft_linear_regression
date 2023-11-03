from predict_price import predict
import numpy as np
from utils import min_max_normalize


class Linear_regression:
    def __init__(self, x, y, predict):
        self.x = x
        self.y = y
        self.norm_predict = predict
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.history = np.array([])
        self.m = float(len(x))
        self.x_min = 0.0
        self.x_max = 0.0
        self.y_min = 0.0
        self.y_max = 0.0
        
    def gradient_descent(self, epoch, lr):
        nm_norm_x, self.x_min, self.x_max = min_max_normalize(self.x)
        nm_norm_y, self.y_min, self.y_max = min_max_normalize(self.y)
        self.history = np.zeros(epoch)
        for i in range(epoch):
            predict_model = self.norm_predict(self.theta0, self.theta1)
            each_diff = predict_model.predict_price(nm_norm_x) - nm_norm_y
            tmp_theta0 = lr / self.m * np.sum(each_diff)
            tmp_theta1 = lr / self.m * np.sum(each_diff * nm_norm_x)
            self.theta0 = self.theta0 - tmp_theta0
            self.theta1 = self.theta1 - tmp_theta1
            print(f"{i} epoch loss: {self.loss(nm_norm_x, nm_norm_y)}")
            self.history[i] = self.loss(nm_norm_x, nm_norm_y)
        return self.theta0, self.theta1
    
    def loss(self, x, y):
        estimate_model = predict(self.theta0, self.theta1)
        each_diff = (estimate_model.predict_price(x) - y) ** 2
        total_loss = sum(each_diff) / self.m
        return total_loss
    
    def predict(self, x):
        predict_model = self.norm_predict(self.theta0, self.theta1)
        norm_x = (x - self.x_min) / (self.x_max - self.x_min)
        norm_y = predict_model.predict_price(norm_x)
        y = norm_y * (self.y_max - self.y_min) + self.y_min
        return y
        
