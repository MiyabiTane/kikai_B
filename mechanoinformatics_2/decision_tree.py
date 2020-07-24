import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as split
import collections as c
import argparse
import math

class Node:

    def __init__(self, data, target, max_depth, target_num, cost_mode):
        self.data = data
        self.target = target
        self.left = None
        self.right = None
        self.feature_i = None
        self.threshold = None
        self.max_depth = max_depth
        self.class_num = len(set(target))
        self.K = target_num
        #その葉ノードに一番多く含まれるクラス
        self.label = c.Counter(target).most_common()[0][0]
        self.mode = cost_mode

    def gini_index(self, T_target):
        #Qm(data) = Σ pmk (1 - pmk)
        Qm = 0
        pm = [0] * self.K
        for tar in T_target:
            pm[tar] += 1
        for i in range(self.class_num):
            pmk = pm[i] / len(T_target)
            Qm += pmk * (1 - pmk)
        return Qm
    
    def cross_entropy(self, T_target):
        #Qm(data) = -Σ pmk ln pmk
        Qm = 0
        pm = [0] * self.K
        for tar in T_target:
            pm[tar] += 1
        for i in range(self.class_num):
            pmk = pm[i] / len(T_target)
            if pmk != 0:
                Qm -= pmk * math.log(pmk)
        return Qm    

    def calc_cost(self, l_target, r_target):
        #ΔQ = Qm(T) - ((Nml/Nm)Qml(T') + (Nmr/Nm)Qmr(T'))
        Nm = len(self.target)
        Nm_l = len(l_target)
        Nm_r = len(r_target)
        if self.mode == 'gini-index':
            Qm = self.gini_index(self.target)
            Qm_l = self.gini_index(l_target)
            Qm_r = self.gini_index(r_target)
        elif self.mode == 'cross-entropy':
            Qm = self.cross_entropy(self.target)
            Qm_l = self.cross_entropy(l_target)
            Qm_r = self.cross_entropy(r_target)
        d_Q = Qm - (Nm_l * Qm_l + Nm_r * Qm_r) / Nm
        return d_Q

    def split(self, depth):
        #葉ノードの全てのパターンが同じクラスに属するor深さの上限に達したら終了
        if self.class_num == 1 or depth == self.max_depth:
            return

        max_d_Q = 0
        for k in range(len(self.data[0])):  # どの特徴量で分割するか
            for th in self.data[:, k]:  # どの値で分割するか
                l_target = self.target[np.where(self.data[:, k] < th)]
                r_target = self.target[np.where(self.data[:, k] >= th)]
                if len(l_target) > 0 and len(r_target) > 0:
                    d_Q = self.calc_cost(l_target, r_target)
                    if d_Q > max_d_Q:
                        best_feat_i = k
                        best_th = th
                        self.feature_i = best_feat_i
                        self.threshold = best_th
                        max_d_Q = d_Q
        print('depth: {}, feature: {}, threshold: {}, label: {}'.format(
            depth, self.feature_i, self.threshold, self.label))

        #再帰的に木を構築
        if self.feature_i:
            left_num = np.where(self.data[:, best_feat_i] < best_th)
            right_num = np.where(self.data[:, best_feat_i] >= best_th)
            self.left = Node(
                self.data[left_num], self.target[left_num], self.max_depth, self.K, self.mode)
            self.right = Node(
                self.data[right_num], self.target[right_num], self.max_depth, self.K, self.mode)
            print('__left__ depth: {}, label: {}'.format(depth+1, self.left.label))
            print('__right__ depth: {}, label: {}'.format(depth+1, self.right.label))
            print("-------")
            #print("left")
            self.left.split(depth + 1)
            #print("right")#
            self.right.split(depth + 1)

    def predict(self, one_data):
        #先端まで達したら
        if (not self.left) and (not self.right):
            return self.label

        if one_data[self.feature_i] < self.threshold:
            return self.left.predict(one_data)
        else:
            return self.right.predict(one_data)


class DecisionTree():

    def __init__(self, max_depth, target_num):
        self.max_depth = max_depth
        self.tree = None
        self.K = target_num

    def fit(self, train_data, target, cost_mode):
        self.tree = Node(train_data, target, self.max_depth, self.K, cost_mode)
        self.tree.split(0)

    def predict(self, test_data):
        ans = [self.tree.predict(one_data) for one_data in test_data]
        return np.array(ans)


def decision_tree_main(cost_mode, max_depth):
    iris = load_iris()
    data = iris.get("data")
    target = iris.get("target")
    train_d, test_d, train_t, test_t = split(
        data, target, test_size=0.3, random_state=0)

    d_tree = DecisionTree(max_depth=int(max_depth), target_num=len(set(target)))
    d_tree.fit(train_d, train_t, cost_mode)
    pred_list = d_tree.predict(test_d)
    #精度計算
    accuracy = 0
    for pred, answer in zip(pred_list, test_t):
        if pred == answer:
            accuracy += 1
    accuracy = float(accuracy / len(pred_list))
    print("max depth : ", max_depth, "accuracy : ", accuracy)
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-mode", choices=['cross-entropy', 'gini-index'])
    parser.add_argument("--max-depth", required=True)
    args = parser.parse_args()
    decision_tree_main(args.cost_mode, args.max_depth)

#main()



