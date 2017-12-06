from itertools import combinations, permutations
import numpy as np

import matplotlib.pyplot as plt


# 指标函数
def cost_maker(distance):
    def cost(path):
        J = sum([ distance[path[i], path[i + 1]] for i in range(len(path) - 1)])+distance[path[-1], path[0]]
        return J
    return cost

# 随机生成函数
def randGen_maker(n):
    def randGen():
        return list(np.random.permutation(n))
    return randGen




# 模拟退火算法
def sa(cost,randGen,iter,**kwargs):
    init_acprop= kwargs['init_acprop'] if 'init_acprop' in kwargs.keys() else None
    tmax= kwargs['tmax'] if 'tmax' in kwargs.keys() else None
    assert(not (init_acprop ==None and  tmax ==None) )
    assert(not (init_acprop !=None and  tmax !=None) )

    # 随机转移：逆序交换
    def randMov(path):
        while True:
            i = int(np.random.rand() * len(path))
            j = int(np.random.rand() * len(path))
            if i == j:
                continue

            i, j = min(i, j), max(i, j)
            return path[:i] + path[i:j + 1][::-1] + path[j + 1:]

    def dcost(new_path, old_path):
        return cost(new_path) - cost(old_path)

    # 如果新解比旧解更优，则接受新解，
    # 否则，按概率接受一个较差的新解
    def accept(new_path, old_path, temperature):
        dJ = dcost(new_path, old_path)
        return 1 if dJ < 0 else np.random.rand() < np.exp(-dJ / temperature)

    def heat(temperature):
        return temperature+10

    # 冷却
    def cooldown(temperature):
        return temperature * 0.99

    # 终止判断
    def isEnd(temperature):
        return temperature < 0.01

    # 温度的平衡判断
    def isBalanced(temperature):
        pass

    # 升温计算
    def calAcprop(path,temperature):
        cnt=0
        for i in range(len(path)**2):
            if accept(path,randMov(path),temperature):
                cnt+=1
        return cnt/(len(path)**2)



    cnt=0
    # 初始化
    path=randGen()
    J=cost(path)


    # 如果未指明初始温度，则由初始接受概率计算得出温度
    if not tmax:
        temperature=1
        # 升温
        while calAcprop(path,temperature)<init_acprop:
            temperature=heat(temperature)
    else:
        temperature=tmax

    xs = []
    ys = []

    while not isEnd(temperature):
        for i in range(iter):
        #while not isBalanced(temperature):
            new_path=randMov(path)
            if accept(new_path,path,temperature):
                cnt=cnt+1
                path=new_path
                J=cost(new_path)

                xs.append(cnt)
                ys.append(J)
                #print(new_path,J,cnt)


        temperature=cooldown(temperature)

    return path,J,cnt,xs,ys




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
    print(distance)

    cost=cost_maker(distance)

    seq = 'ACLBIQFTMEPRGSOJHDKN'
    best_path = [ord(c) - ord('A') for c in seq]
    print(best_path, cost(best_path))
    print()
    # [0, 2, 11, 1, 8, 16, 5, 19, 12, 4, 15, 17, 6, 18, 14, 9, 7, 3, 10, 13] 41.914474

    randGen=randGen_maker(n)

    # 280 400
    # 40.836798000000002

    #tmax
    #init_acprop
    path,J,cnt,xs,ys= sa(cost=cost,randGen=randGen,tmax=10,iter=50)
    print(path,J,cnt)
    print(np.math.factorial(20))


    plt.plot(xs, ys)
    plt.show()
    pass
    # 50 10
    #{40.836798000000016: 7, 40.836798000000002: 27, 41.359370000000006: 12, 42.357352000000013: 3, 41.359369999999998: 4, 40.836798000000009: 26, 40.836797999999995: 8, 41.359370000000013: 11, 42.357352000000006: 1, 42.357351999999999: 1}

    #{41.359370000000006: 8, 40.836798000000002: 21, 40.836798000000009: 22, 40.836798000000016: 7,
    # 40.836797999999995: 17, 41.359370000000013: 10, 41.35937000000002: 2, 42.357352000000006: 4, 42.357351999999999: 4,
    # 41.359369999999998: 4, 42.357352000000013: 1}

    Jd=dict()
    for i in range(100):
        _,J,__,___,____=sa(cost=cost, randGen=randGen, init_acprop=0.8, iter=50)
        Jd[J]=Jd.setdefault(J,0)+1
        print(J,__)

    print(Jd)




