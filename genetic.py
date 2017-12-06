from itertools import permutations, combinations
from scipy.special import comb
import numpy as np

import matplotlib.pyplot as plt


def printPop(pops):
    print("scale:",len(pops))
    for unit in pops:
        print(unit)

# 适应函数
def fitness_fun_maker(distance):
    # 取总长度的倒数
    def fitness_fun(path):
        J = sum([distance[path[i], path[i + 1]] for i in range(len(path) - 1)]) + distance[path[-1], path[0]]
        return 1 / J

    return fitness_fun

# 随机生成函数
def randGen_maker(n):
    def randGen():
        return list(np.random.permutation(n))

    return randGen


# 遗传算法
def ga(fitness_fun,randGen,scale,cross_prop,mutation_prop,iter):
    # 初始化：随机生成种群
    def initializeGen(scale):
        return [randGen() for i in range(scale)]

    # 选择：轮盘法
    def selection(props, pops):
        rev = []
        for i in range(len(pops)):
            threshold = np.random.rand()
            si = -1
            select_value = 0
            while select_value < threshold:
                si += 1
                select_value += props[si]
            rev.append(pops[si])
        return rev

    # 交叉：由父代产生子代，常规交配法
    def next_generation(father, mother):
        pos = int(np.random.rand() * len(father))
        son =father[:pos]+list(filter(lambda gene:gene not in father[:pos] ,mother))
        daughter=mother[:pos]+list(filter(lambda gene:gene not in mother[:pos] ,father))
        return son,daughter

    # 交叉：种群内按概率两两配对
    def cross(pops):
        for father_i, mother_j in combinations(range(len(pops)), 2):
            if np.random.rand() < cross_prop:
                pops[father_i], pops[mother_j] = next_generation(pops[father_i], pops[mother_j])
        return pops


    # 变异：逆序交换
    def self_muta(unit):
        if np.random.rand() < mutation_prop:
            while True:
                i = int(np.random.rand() * len(unit))
                j = int(np.random.rand() * len(unit))
                if i == j:
                    continue

                i, j = min(i, j), max(i, j)
                return unit[:i] + unit[i:j + 1][::-1] + unit[j + 1:]
        return unit

    def mutation(pops):
        return list(map(self_muta, pops))

    # 每个配对的交叉概率为全体交叉概率的2/(n-1)
    cross_prop=cross_prop*2/(scale-1)

    # 初始化种群
    pops=initializeGen(scale)

    xs=range(iter)
    ys=[]

    # 进化迭代
    for it in range(iter):
        fits=[ fitness_fun(unit) for unit in pops]
        props=fits/sum(fits)
        pops=selection(props,pops)
        pops=cross(pops)
        pops=mutation(pops)

        # 每次迭代记录最优值
        maxi = np.argmax(fits)
        ys.append(fits[maxi])

    # 取最大值返回
    maxi=np.argmax(fits)
    return pops[maxi],1/fits[maxi],xs,ys





if __name__ == '__main__':
    n = 20
    citys = np.array([
        [5.294, 1.558],
        [4.286, 3.622],
        [4.719, 2.774],
        [4.185, 2.230],
        [0.915, 3.821],
        [4.771, 6.041],
        [1.524, 2.871],
        [3.447, 2.111],
        [3.718, 3.665],
        [2.649, 2.556],
        [4.399, 1.194],
        [4.660, 2.949],
        [1.232, 6.440],
        [5.036, 0.244],
        [2.710, 3.140],
        [1.072, 3.454],
        [5.855, 6.203],
        [0.194, 1.862],
        [1.762, 2.693],
        [2.682, 6.097]
    ])
    distance = np.zeros(shape=(n, n))
    for i, j in permutations(range(0, n), 2):
        distance[i, j] = np.sum((citys[i] - citys[j]) ** 2, axis=0, keepdims=True)
    # print(distance)

    fitness_fun=fitness_fun_maker(distance)
    randGen=randGen_maker(n)
    _,__,xs,ys=ga(fitness_fun=fitness_fun,randGen=randGen,scale=100,cross_prop=0.1,mutation_prop=0.02,iter=10000)

    print(_,__)

    plt.plot(xs,ys)
    plt.show()
