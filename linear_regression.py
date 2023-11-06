import numpy as np

from utils import min_max_normalize
from predict import predict_price

class Linear_regression:
    def __init__ (self, x, y):
        self.x = x
        self.y = y
        self.theta0, self.theta1 = 0.0, 0.0
        self.theta_history = []
        self.loss_history = []
        self.x_min, self.x_max = 0.0, 0.0
        self.y_min, self.y_max = 0.0, 0.0
        
    def train(self, epochs, lr):
        norm_x, self.x_min, self.x_max = min_max_normalize(self.x)
        norm_y, self.y_min, self.y_max = min_max_normalize(self.y)
        self.theta0, self.theta1 = 0.0, 0.0
        tmp_theta0 = 0.0
        tmp_theta1 = 0.0
        for epoch in range(0, epochs):
            diff = self.norm_predict(norm_x) - norm_y
            tmp_theta0 = lr / float(len(self.x)) * np.sum(diff)
            tmp_theta1 = lr / float(len(self.x)) * np.sum(diff * norm_x)
            self.theta_history.append([self.theta0, self.theta1])
            loss = sum(diff ** 2) / float(len(self.x))
            self.loss_history.append(loss)
            self.theta0 -= tmp_theta0
            self.theta1 -= tmp_theta1
            print(f"{self.theta0} {self.theta1}")
            print(f"{epoch} epoch loss: {loss}")
            
            
            
    def norm_predict(self, x):
        return self.theta0 + self.theta1 * x
    
    def predict(self, x):
        x_z = (x - self.x_min) / (self.x_max - self.x_min)
        y_z = self.theta0 + self.theta1 * x_z
        y = (self.y_max - self.y_min) * y_z + self.y_min
        return y
