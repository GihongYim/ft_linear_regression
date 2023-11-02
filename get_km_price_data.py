import pandas as pd


def get_km_price_data(filename):
    data = pd.read_csv(filename)
    data['km'] = data['km'].astype('float')
    data['price'] = data['price'].astype('float')
    print(data)
    return data
