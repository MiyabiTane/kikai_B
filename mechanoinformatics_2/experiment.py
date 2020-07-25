from decision_tree import decision_tree_main as main
import matplotlib.pyplot as plt
import os

def experiment():
    accuracy = [[], []]
    for i in range(2):
        for j in range(0, 6):
            cost_mode = "gini-index" if i == 0 else "cross-entropy"
            acc = main(cost_mode=cost_mode, max_depth=j)
            accuracy[i].append(acc)
    print(accuracy)
    return accuracy

def vis_graph(accuracy):
    os.makedirs("img", exist_ok=True)
    labels = ["gini index", "cross entropy"]
    x = [i for i in range(6)]
    plt.figure()
    for i in range(2):
        plt.xlabel("max depth")
        plt.ylabel("accuracy")
        plt.plot(x, accuracy[i], label = labels[i])
    plt.title("accuracy")
    plt.legend()
    #plt.show()
    plt.savefig("img/accuracy.png")

accuracy = experiment()
vis_graph(accuracy)
