from hopfield import Hopfield
from hopfield import show_result
from hopfield import add_noise
from hopfield import evaluate_sim
from hopfield import evaluate_acc
import matplotlib.pyplot as plt
import numpy as np
import os
from hopfield import Hopfield
from hopfield import show_result
from hopfield import add_noise
from hopfield import evaluate_sim
from hopfield import evaluate_acc
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def make_trains(size = 6):
    train_matrixs = []
    for _ in range(size):
        train_mat = []
        for _ in range(25):
            rand = random.randint(0, 1)
            if rand == 0:
                train_mat.append(-1)
            else:
                train_mat.append(1)
        train_matrixs.append(train_mat)
    return train_matrixs


def for_test(train_matrixs, output_dir, noise_max, noise_step):
    sim = []
    acc = []
    for noise in range(5, noise_max + 1, noise_step):
        sim_sub = []
        acc_sub = []
        print("noise {}%".format(noise))
        #各条件に対して10回試行して平均をとる
        for _ in range(0, 10):
            test_matrix = add_noise(train_matrixs[0], noise)
            hop = Hopfield(train_matrixs, test_matrix, max_time=100)
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
def test_op2(train_matrixs):
    os.makedirs("graphs", exist_ok=True)
    sim_list = []
    acc_list = []
    for i in range(1, 7):
        new_train_matrixs = train_matrixs[:i].copy()
        sim, acc = for_test(new_train_matrixs, "random_inputs/" +
                            str(i) + "_matrixs", 20, 1)
        sim_list.append(sim)
        acc_list.append(acc)
    plt.figure()
    for i, y_acc in enumerate(acc_list):
        x_list = [i for i in range(5, 21)]
        plt.plot(x_list, y_acc, label="input" + str(i + 1))
        plt.xlabel("noise[%]")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("accuracy")
        plt.savefig("graphs/op2_accuracy.png")
    plt.figure()
    for i, y_sim in enumerate(sim_list):
        x_list = [i for i in range(5, 21)]
        plt.plot(x_list, y_sim, label="input" + str(i + 1))
        plt.xlabel("noise[%]")
        plt.ylabel("simularity")
        plt.legend()
        plt.title("simularity")
        plt.savefig("graphs/op2_simularity.png")

train_matrixs = np.array(make_trains())
test_op2(train_matrixs)
