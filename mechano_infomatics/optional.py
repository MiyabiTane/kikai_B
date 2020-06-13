from hopfield import Hopfield
from hopfield import show_result
from hopfield import add_noise
from hopfield import evaluate_sim
from hopfield import evaluate_acc
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

train_matrixs = np.array([[-1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1,
                           -1, 1, 1, 1, -1, -1, 1, -1, 1, -1],
                          [1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1,
                           1, -1, -1, 1, -1, 1, 1, 1, -1, -1],
                          [-1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1,
                           -1, 1, -1, -1, -1, -1, 1, 1, 1, -1],
                          [1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1,
                           1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
                          [-1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1,
                           -1, 1, -1, -1, -1, -1, 1, 1, 1, -1],
                          [-1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1,
                           1, -1, -1, -1, 1, 1, 1, 1, 1, 1]])


def for_test_op(train_matrixs, output_dir, noise_max, noise_step, thresh):
    sim = []
    acc = []
    for noise in range(5, noise_max + 1, noise_step):
        sim_sub = []
        acc_sub = []
        print("noise {}%".format(noise))
        #各条件に対して10回試行して平均をとる
        for _ in range(0, 10):
            test_matrix = add_noise(train_matrixs[0], noise)
            hop = Hopfield(train_matrixs, test_matrix, max_time=100, threshold = [thresh]*25)
            output = hop.predict()
            print("check")
            simularity = evaluate_sim(train_matrixs, output)
            accurate = evaluate_acc(train_matrixs, output)
            print("accurate: ", accurate, " simularity: ", simularity)
            sim_sub.append(simularity)
            acc_sub.append(accurate)
        sim.append(np.mean(np.array(sim_sub)))
        acc.append(np.mean(np.array(acc_sub)))
        #最後の１回を図示
        print("show")
        os.makedirs(output_dir, exist_ok=True)
        show_result(train_matrixs, test_matrix, output,
                    output_dir + "/" + str(noise) + ".png")
    print(sim)
    print(acc)
    return sim, acc


#画像の種類を６まで増やしてみる
def test_optional(train_matrixs):
    os.makedirs("graphs", exist_ok=True)
    sim_list = []
    acc_list = []
    for i in range(-10, 11, 5):
        sim, acc = for_test_op(train_matrixs, "thresh/" +
                            str(i) + "_thresh", 20, 1, i)
        sim_list.append(sim)
        acc_list.append(acc)
    plt.figure()
    for i, y_acc in enumerate(acc_list):
        x_list = [i for i in range(5, 21)]
        plt.plot(x_list, y_acc, label="threshold=" + str((i - 2) * 5))
        plt.xlabel("noise[%]")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("accuracy")
        plt.savefig("graphs/optional_accuracy.png")
    plt.figure()
    for i, y_sim in enumerate(sim_list):
        x_list = [i for i in range(5, 21)]
        plt.plot(x_list, y_sim, label="threshold" + str((i - 2) * 5))
        plt.xlabel("noise[%]")
        plt.ylabel("simularity")
        plt.legend()
        plt.title("simularity")
        plt.savefig("graphs/optional_simularity.png")


test_optional(train_matrixs[:1])