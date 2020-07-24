import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as split

def visualize(test_data, test_target, mode):
    #petal length, petal widthを軸にとって可視化
    markers = ["p", "*", "o"]
    labels = ["Setosa", "Versicolor", "Virginica"]
    plt.figure()
    for i in range(len(set(test_target))):
        data = test_data[np.where(test_target == i)]
        x1 = data[:, 2]
        x2 = data[:, 3]
        plt.xlabel("Petal length [cm]")
        plt.ylabel("Petal width [cm]")
        plt.scatter(x1, x2, marker=markers[i], label=labels[i])
    plt.ylim(0, 3)
    plt.xlim(0, 7)
    #ターミナルの出力結果を利用して作成した
    #max_depth=3の図示
    if mode == "gini-index":
        plt.axvspan(0, 3, alpha=0.3, color="yellow")
        plt.axvspan(3, 5, 0, 1.7/3, alpha=0.3, color="red")
        plt.axvspan(5, 7, alpha=0.3, color="blue")
        plt.axvspan(3, 5, 1.7/3, 1, alpha=0.3, color="blue")
        plt.axvline(3, color="blue")
        plt.axvline(5, color="green")
        plt.axvline(5.1, color="red")
        plt.hlines(y=1.7, xmin=3, xmax=5, color="red")
        plt.title("Classify by DecisionTree (gini index)")
        plt.legend()
        plt.savefig("img/gini-index.png")
        #plt.show()
    if mode == "cross-entropy":
        plt.axvspan(0, 3, alpha=0.3, color="yellow")
        plt.axvspan(3, 5.1, 0, 1.7/3, alpha=0.3, color="red")
        plt.axvspan(5.1, 7, alpha=0.3, color="blue")
        plt.axvspan(3, 5.1, 1.7/3, 1, alpha=0.3, color="blue")
        plt.axvline(3, color="blue")
        plt.axvline(5.1, color="green")
        plt.hlines(y=1.7, xmin=3, xmax=5.1, color="red")
        plt.title("Classify by DecisionTree (cross entropy)")
        plt.legend()
        plt.savefig("img/cross-entropy.png")
        #plt.show()


iris = load_iris()
data = iris.get("data")
target = iris.get("target")
train_d, test_d, train_t, test_t = split(
    data, target, test_size=0.3, random_state=0)
visualize(test_d, test_t, "gini-index")
visualize(test_d, test_t, "cross-entropy")
