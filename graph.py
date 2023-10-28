import matplotlib.pyplot as plt
import seaborn as sns

def show_data_graph(data):
    plt.figure()
    sns.scatterplot(data, x=data["km"], y=data["price"])
    plt.show()