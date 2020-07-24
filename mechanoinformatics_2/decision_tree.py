import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as split
import collections as c

class Node:

    def __init__(self, data, target, max_depth, ):
        self.data = data
        self.target = target
        self.left = None
        self.right = None
        self.feature_i = None
        self.threshold = None
        self.max_depth = max_depth
        self.K = len(set(target))
        #その葉ノードに一番多く含まれるクラス
        self.label = c.Counter(target).most_common()[0][0]
        

    def gini_index(self, T_target):
    #Qm(data) = Σpmk (1 - pmk)
        Qm = 0
        pm = [0, 0, 0] 
        for tar in T_target:
            pm[tar] += 1
        for i in range(self.K):
            pmk = pm[i] / len(T_target)
            Qm += pmk * (1 - pmk)
        return Qm


    def calc_cost(self, l_target, r_target):
    #ΔQ = Qm(T) - ((Nml/Nm)Qml(T') + (Nmr/Nm)Qmr(T')) 
        Nm = len(self.target)
        Nm_l = len(l_target)
        Nm_r = len(r_target)
        Qm = self.gini_index(self.target)
        Qm_l = self.gini_index(l_target)
        Qm_r = self.gini_index(r_target)
        d_Q = Qm - (Nm_l * Qm_l + Nm_r * Qm_r) / Nm
        return d_Q
    

    def split(self, depth):
        #葉ノードの全てのパターンが同じクラスに属するor深さの上限に達したら終了
        if self.K == 1 or depth == self.max_depth:
            return 

        max_d_Q = 0
        for k in range(3): #どの特徴量で分割するか
            for th in self.data[:, k]: #どの値で分割するか
                l_target = self.target[np.where(self.data[:, k] < th)]
                r_target = self.target[np.where(self.data[:, k] >= th)]
                #print("left : ", l_target)
                #print("right :", r_target)
                if len(l_target) > 0 and len(r_target) > 0:
                    d_Q = self.calc_cost(l_target, r_target)
                    if d_Q > max_d_Q:
                        best_feat_i = k; best_th = th
                        self.feature_i = best_feat_i
                        self.threshold = best_th
                        max_d_Q = d_Q
        print('depth: {}, feature: {}, threshold: {}, label: {}'.format(depth, self.feature_i, self.threshold, self.label))
        
        #再帰的に木を構築
        left_num = np.where(self.data[:, best_feat_i] < best_th)
        right_num = np.where(self.data[:, best_feat_i] >= best_th)
        self.left = Node(self.data[left_num], self.target[left_num], self.max_depth)
        self.right = Node(self.data[right_num], self.target[right_num], self.max_depth)
        self.left.split(depth + 1)
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

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, train_data, target):
        self.tree = Node(train_data, target, self.max_depth)
        self.tree.split(0)
    
    def predict(self, test_data):
        ans = [self.tree.predict(one_data) for one_data in test_data]
        return np.array(ans)


#メイン文
iris = load_iris()
data = iris.get("data")
target = iris.get("target")
train_d, test_d, train_t, test_t = split(data, target, test_size=0.3, random_state=0)

d_tree = DecisionTree(max_depth = 4)
d_tree.fit(train_d, train_t)
pred_list = d_tree.predict(test_d)
#精度計算
accuracy = 0
for pred, answer in zip(pred_list, test_t):
    if pred == answer:
        accuracy += 1
accuracy = float(accuracy / len(pred_list))
print("accuracy : ", accuracy)


        

