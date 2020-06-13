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
        show_result(train_matrixs, test_matrix, output, output_dir + "/" + str(noise) + ".png")
    print(sim)
    print(acc)
    return sim, acc


#１種類の画像で調べる
def test_1(train_matrixs):
    os.makedirs("graphs", exist_ok=True)
    sim, acc = for_test(train_matrixs, "one_matrix", 20, 1)
    plt.figure()
    x_list = [i for i in range(5, 21)]
    plt.plot(x_list, sim, label="simularity")
    plt.plot(x_list, acc, label="accuracy")
    plt.xlabel("noise[%]")
    plt.legend()
    plt.title("input one matrix")
    plt.savefig("graphs/one_matrix")


#画像の種類を６まで増やしてみる
def test_2(train_matrixs):
    os.makedirs("graphs", exist_ok = True)
    sim_list = []
    acc_list = []
    for i in range(1,7):
        new_train_matrixs = train_matrixs[:i].copy()
        sim, acc = for_test(new_train_matrixs, "one_six/" + str(i) + "_matrixs", 20, 1)
        sim_list.append(sim)
        acc_list.append(acc)
    plt.figure()
    for i, y_acc in enumerate(acc_list):
        x_list = [i for i in range(5, 21)]
        plt.plot(x_list, y_acc, label = "input" + str(i + 1))
        plt.xlabel("noise[%]")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("accuracy")
        plt.savefig("graphs/accuracy.png")
    plt.figure()
    for i, y_sim in enumerate(sim_list):
        x_list = [i for i in range(5, 21)]
        plt.plot(x_list, y_sim, label = "input" + str(i + 1))
        plt.xlabel("noise[%]")
        plt.ylabel("simularity")
        plt.legend()
        plt.title("simularity")
        plt.savefig("graphs/simularity.png")

#ノイズを５％ごとに0から100まで増やす
def test_3(train_matrixs):
    os.makedirs("graphs", exist_ok=True)
    sim_list = []
    acc_list = []
    new_train_matrixs = train_matrixs[:2].copy()
    sim, acc = for_test(new_train_matrixs, "noise_100/two_matrixs", 100, 5)
    sim_list.append(sim)
    acc_list.append(acc)
    new_train_matrixs = train_matrixs[:4].copy()
    sim, acc = for_test(new_train_matrixs, "noise_100/four_matrixs", 100, 5)
    sim_list.append(sim)
    acc_list.append(acc)
    plt.figure()
    for i, y_acc in enumerate(acc_list):
        x_list = [i for i in range(5, 101, 5)]
        plt.plot(x_list, y_acc, label="input" + str((i + 1) * 2))
        plt.xlabel("noise[%]")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("accuracy")
        plt.savefig("graphs/test_3_accuracy.png")
    plt.figure()
    for i, y_sim in enumerate(sim_list):
        x_list = [i for i in range(5, 101, 5)]
        plt.plot(x_list, y_sim, label="input" + str((i + 1) * 2))
        plt.xlabel("noise[%]")
        plt.ylabel("simularity")
        plt.legend()
        plt.title("simurality")
        plt.savefig("graphs/test_3_simularity.png")


parser = argparse.ArgumentParser()
parser.add_argument("--test", choices = ["test1", "test2", "test3"])
args = parser.parse_args()

if args.test == "test1":
    test_1(train_matrixs[:1])
elif args.test == "test2":
    test_2(train_matrixs)
elif args.test == "test3":
    test_3(train_matrixs)


    
