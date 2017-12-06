


# None代表可任意适配
# fit 判断 概念A和 概念B 是否适配
def fit(conceptA,conceptB):
    if conceptA==None or conceptB==None or conceptA==conceptB:
        return True
    else:
        return False


# 检查 两个多元组 是否适配
def checkEle(eleA,eleB):
    for a,b in zip(eleA,eleB):
        if not fit(a,b):
            return False
    return True


# 一次泛化操作
# 例如： 把 'Y', '△', 'soft', 'large'
# 泛化为
# 'None', '△', 'soft', 'large'
# 'Y', 'None', 'soft', 'large'
# 'Y', '△', 'None', 'large'
# 'Y', '△', 'soft', 'None'
def generalize(space,ele):
    returnSet=set()
    for pi in range(len(ele)):
        if not ele[pi]==None:
            spele=list(ele)
            spele[pi] = None
            returnSet.add(tuple(spele))
    return returnSet





# 一次特化操作
# 例如： 把 None,None,None,None
# 特化为
# (None, None, 'hard', None),
# ('Y', None, None, None),
# 等等
def specialize(space,ele):
    returnSet=set()
    for pi in range(len(ele)):
        if ele[pi]==None:
            spele=list(ele)
            for prop in space[pi]:
                spele[pi]=prop
                returnSet.add(tuple(spele))
    return returnSet



# 正例学习
def posLearn(space,data,G,S):
    # 此处可用filter()
    # 删除G中不与正例适配的
    for ele in G.copy():
        if not checkEle(ele, data):
            G.remove(ele)

    #对S进行修改，做尽量少的泛化，并使之覆盖正例
    Scopy = S.copy()
    while True:
        Srestore = set()
        for ele in Scopy:
            if not checkEle(ele, data):
                if ele in S:
                    S.remove(ele)
                for genele in generalize(space, ele):
                    Srestore.add(genele)
                    if checkEle(genele, data):
                        S.add(genele)

        # 如果一次泛化后S为空，就要再进行泛化
        if S:
            break
        Scopy.update(Srestore)

    return G,S






# 反例学习
def negLearn(space,data,G,S):
    # 此处可用filter()
    # 删除S中与反例适配的
    for ele in S.copy():
        if checkEle(ele, data):
            S.remove(ele)

    #对G进行修改，做尽量少的特化，并使之不覆盖反例
    Gcopy = G.copy()
    while True:
        Grestore = set()

        for ele in Gcopy:
            if checkEle(ele, data):
                if ele in G:
                    G.remove(ele)
                for spele in specialize(space, ele):
                    Grestore.add(ele)
                    if not checkEle(spele, data):
                        G.add(spele)

        # 如果一次特化后G为空，就要再进行特化
        if G:
            break
        Gcopy.update(Grestore)

    return G,S


def learn(space,train,S,G= {(None,None,None,None)}):
    for data,label in train:
        print('input:',data,label)

        # 正例
        if label:
            G,S=posLearn(space,data,G,S)
        # 反例
        else:
            G, S = negLearn(space, data, G, S)

        print('G:',G)
        print('S:',S)
        print()

    # 结果返回
    return G,S



if __name__ == '__main__':
    # 概念空间
    color = ['Y', 'B', 'G']
    shape = ['△', '○', '□']
    solidity = ['hard', 'soft']
    size = ['large', 'small']
    space = [color, shape, solidity, size]

    # G初始化为全适配
    G = set([(None, None, None, None)])
    # S用第一个正例初始化
    S = set([('Y', '△', 'soft', 'large')])

    # 训练集合
    train = [
        (('B', '□', 'soft', 'small'), False),
        (('Y', '○', 'soft', 'small'), True),
        (('Y', '△', 'hard', 'large'), True),
    ]

    #　学习Y
    learn(space,train,S,G)



