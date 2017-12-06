import numpy as np
from numpy import *






# x: data
# w: para
# b: what
# z = w^T·x + b


# a:y_hat
# a= σ(z)
# y:label
# L: loss L(a,y)= -(y*log(a)+(1-y)*log(1-a))
# J: 1/m * sigma(L)[for i=1 to m]

# dL_da
# dL_dz = (dL_da* da_dz) = a - y

# dL_dw = x * dL_dz
# dL_db = dL_dz


def sigmoid(x):
    return 1/(1+np.exp(-x))


def loss(a,y):
    return -(y*np.log(a)+(1-y)*np.log(1-a))

def logR(n,m,alpha):

    w=np.random.rand(n,1)
    # 数据-x标号
    x=np.random.rand(n,m)
    b=np.random.rand(1)
    y=sigmoid(np.random.rand(1,m))
    print(w)




    for i in range(0,1000):
        # frontward
        # auto transpos w
        z = np.dot(w.T, x) + b
        a = sigmoid(z)


        #J=1/m*np.sum(loss(a,y))

        # backward
        dJ_dz=a-y

        dJ_dw=1/m*np.dot(x,dJ_dz.T)
        dJ_db=1/m*np.sum(dJ_dz)

        # down
        w=w-alpha*dJ_dw
        b=b-alpha*dJ_db

    return w,x,b,y






if __name__ == '__main__':
    n,m=2,5
    w,x,b,y=logR(n,m,alpha=0.1)
    yh = sigmoid(np.dot(w.T, x) + b)
    cnt = 0
    for i in range(0, m):
        if abs(y[0, i] - yh[0, i]) > 0.01:
            cnt += 1

    print((m - cnt) / m)