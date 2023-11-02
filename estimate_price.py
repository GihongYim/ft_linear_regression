class estimate:
    def __init__(self, theta0, theta1):
        self.theta0 = theta0
        self.theta1 = theta1

    def estimate_price(self, mileage):
        return self.theta0 + self.theta1 * mileage
