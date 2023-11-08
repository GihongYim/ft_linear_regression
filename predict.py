from linear_regression import Linear_regression
import pickle
def predict():
    theta0 = 0.0
    theta1 = 0.0
    model = Linear_regression()
    try:
        with open('parameter.pickle', 'rb') as file:
            theta0 = pickle.load(file)
            theta1 = pickle.load(file)
            x_min = pickle.load(file)
            x_max = pickle.load(file)
            y_min = pickle.load(file)
            y_max = pickle.load(file)
            model.theta0 = theta0
            model.theta1 = theta1
            model.x_min = x_min
            model.x_max = x_max
            model.y_min = y_min
            model.y_max = y_max
    except Exception as e:
        theta0 = 0.0
        theta1 = 0.0
        print(f"parameter.pickle not found theta0 theta1 set 0.0")
        mileage = int(input("Enter your car's mileage: "))
        print('0.0')
        exit(0)
    mileage = int(input("Enter your car's mileage: "))
    km = model.predict(mileage)
    return km
if __name__ == "__main__":
    print(predict())