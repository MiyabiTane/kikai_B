import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


class Hopfield:

    def __init__(self, train_matrixs, test_matrix, max_time=1000, threshold=[0]*25):
        """
        train_matrixs = [train_mat1, train_mat2, ...] :np.array
        train_mat :np.array
        threshold.shape = (25,)
        test.matrix :np.array
        """
        #self.ifif = np.array(train_matrixs[0])
        self.original_matrix = train_matrixs[0]
        self.threshold = threshold
        self.max_time = max_time
        self.test_matrix = test_matrix.copy()
        print(self.test_matrix)
        #記憶する画像の数
        self.Q = len(train_matrixs)
        #記憶する画像の大きさ。ここでは25
        self.img_size = len(train_matrixs[0])

        #重みの初期化 W = (Σ xq・xq.T) / Q (ここではxq = xq.T)
        self.w = np.zeros((self.img_size, self.img_size), dtype=np.float)
        for i in range(self.Q):
            self.w += np.dot(train_matrixs[i].reshape(self.img_size, 1),
                             train_matrixs[i].reshape(1, self.img_size))
        for i in range(self.img_size):
            self.w[i][i] = 0
        self.w /= self.Q
        #print(np.array(self.w))

    def energy(self, matrix):
        """
        matrix.shape = (1, 25)

        V = -1/2ΣΣwijxixj + Σθixi
        xi = matrix[i]
        """
        V = 0
        for i in range(self.img_size):
            V += self.threshold[i] * matrix[i]
            for j in range(self.img_size):
                V += self.w[i][j] * matrix[i] * matrix[j]
        return -0.5 * V

    def sgn(self, x):
        return -1 if x < 0 else 1

    def update_x(self, i):
        """
        xi(t+1) = f(Σwijxj(t))
        f(u) = sgn(u-θi)
        """
        u = 0
        for j in range(self.img_size):
            u += self.w[i][j] * self.test_matrix[j]
        return self.sgn(u - self.threshold[i])


    def update(self):
        for t in range(self.max_time):
            V_pre = self.energy(self.test_matrix)
            new_test_matrix = self.test_matrix.copy()
            for i in range(self.img_size):
                new_test_matrix[i] = self.update_x(i)
            V_aft = self.energy(new_test_matrix)
            self.test_matrix = new_test_matrix.copy()
            print(t, " : ", V_aft - V_pre)
            if abs(V_aft - V_pre) == 0:
                break


    def predict(self):
        self.update()
        self.test_matrix = self.test_matrix.reshape(5, 5)
        return self.test_matrix


def add_noise(data, percent):
    """
    data.shape = (25,) :np.array
    """
    copy = data.copy()
    noise_num = np.array([random.randint(0, 100) for i in range(25)])
    #値反転
    copy[np.where(noise_num < percent)] *= -1
    return copy


def evaluate_sim(train_matrixs, output):
    max_sim = 0
    for j in range(len(train_matrixs)):
        similarity = 0
        for i in range(25):
            similarity += train_matrixs[j].reshape(25,)[i] * output.reshape(25,)[i]
        candidate = similarity / 25
        if candidate > max_sim:
            max_sim = candidate
    return max_sim


def evaluate_acc(train_matrixs, output):
    for i in range(len(train_matrixs)):
        if not False in (train_matrixs[i].reshape(25,) == output.reshape(25,)):
                return 1
    return 0


def show_result(trains, test, output, save_name):
    def show(data, index, name):
        copy = data.reshape(5, 5).copy()
        print(copy)
        copy[np.where(copy == -1)] = 255
        copy[np.where(copy == 1)] = 0
        plt.subplot(3, 3, index)
        plt.title(name)
        plt.imshow(copy, cmap="gray")
    for i in range(len(trains)):
        show(trains[i], i+1, "train" + str(i))
    show(test, 7, "test")
    show(output, 8, "output")
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_name)


