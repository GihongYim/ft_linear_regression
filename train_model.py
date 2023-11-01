def train(data, num_of_epoch, learning_rate):
    """_summary_

    Args:
        data (np.DataFrame): x0, x1, x2, x3, ... data, y output data
        num_of_epoch (int): num of train
        learning_rate (_type_): learning rate lr : move parameter rate

    Returns:
        _type_: _description_
    """
    theta0, theta1 = 0.0, 0.0
    for _ in range(num_of_epoch):
        tmp_theta0, tmp_theta1 = 0.0, 0.0
        print(len(data))
        for row_index in range(0, len(data)):
            loss = 0.0
            row = data.loc[row_index]
            print(theta0, theta1, row['km'])
            print("e: ", estimate_price(theta0, theta1, float(row['km'])))
            print("price: ", row['price'])
            loss = estimate_price(theta0, theta1, float(row['km']) / 1000.0) - float(row['price']) / 1000.0
            print("loss: ", loss)
            tmp_theta0 = tmp_theta0 + loss
            tmp_theta1 = tmp_theta1 + (loss * float(row['km']) / 1000)
        theta0 = theta0 - learning_rate / float(len(data)) * tmp_theta0
        theta1 = theta1 - learning_rate / float(len(data)) * tmp_theta1
        print(theta0)
        print(theta1)
    return theta0, theta1


def estimate_price(theta0, theta1, mileage):
    print(f"theta1 * mileage{theta1 * float(mileage)}")
    return theta0 + (theta1 * float(mileage))
