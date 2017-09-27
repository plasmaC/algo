from collections import deque
from functools import reduce
from queue import Queue

import heapq
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image



# 强制numpy打印所有数据
set_printoptions(threshold=np.NaN)


# 起始点


startMx=array([[2,8,3],
              [1,6,4],
              [7,0,5]])

# 终点
endMx=array([[1,2,3],
              [8,0,4],
              [7,6,5]])



# 阻塞地块 set
blockedSet = {}

# 地图
edigPosMap=array([[8,7,6],
                 [5,4,3],
                 [2,1,0]])

# 移动向量
move_v = ((0, 1), (0, -1), (1, 0), (-1, 0))


# 记录零位置
num_zpDict = dict()


# 从矩阵到数值
def load8dig(edigMx)->int:
    factor=1
    edigNum=0
    for i in reversed(range(0,3)):
        for j in reversed(range(0,3)):
            if edigMx[i,j]==0:
                zero_pos=i,j
            edigNum+=edigMx[i,j]*factor
            factor*=10
    return edigNum,zero_pos


# 从数值到矩阵
def show8dig(edigNum)->np.array:
    edigMx=zeros(shape=(3,3),dtype=int)
    for i in reversed(range(0,3)):
        for j in reversed(range(0,3)):
            edigMx[i,j]=edigNum%10
            # numpy中 如果类型为int 则执行地板除，类型为float 则执行精确除
            edigNum/=10
    return edigMx




startMx=array([[2,8,3],
              [1,6,4],
              [7,0,5]])

startNum, szero_pos=load8dig(startMx)
num_zpDict[startNum] = szero_pos

endMx=array([[1,2,3],
              [8,0,4],
              [7,6,5]])


endNum, ezero_pos=load8dig(endMx)



# 获取某一位数值
def getDig(source,npos):
    return source//pow(10,npos)%10


# 交换两个卡牌的位置
def getTraNum(source:int,mpos1:tuple,mpos2:tuple)->int:

    npos1,npos2=edigPosMap[mpos1],edigPosMap[mpos2]

    # 地板除//   精确除/
    dig1,dig2=getDig(source,npos1),getDig(source,npos2)

    # 掉转
    resNum=source+(dig2-dig1)*pow(10,npos1)+(dig1-dig2)*pow(10,npos2)
    return resNum








# 返回上下左右合法的位置
def getNeighbour(sourceNum):
    zero_pos=num_zpDict[sourceNum]
    for move_vj in move_v:
        # 不越界判断
        def isInBounds(point):
            for i in range(len(edigPosMap.shape)):
                if point[i] < 0 or point[i] >= edigPosMap.shape[i]:
                    return False
            return True
        # 障碍判断
        def isOnBlocked(point):
            return point in blockedSet


        nezero_pos = tuple(zero_pos[i] + move_vj[i] for i in range(len(zero_pos)))

        # 准入判断
        judgeBox = isInBounds(nezero_pos), \
                   not isOnBlocked(nezero_pos)
        if judgeBox == (1, 1):
            neNum=getTraNum(sourceNum, zero_pos, nezero_pos)

            # 更新注册
            num_zpDict[neNum]=nezero_pos
            yield neNum








 # 构造最佳路径
def buildRes(printPath, openDict, closeDict, nid_gDict, pathDict):
    # 路径构造和绘制
    path = deque()
    curNum = endNum
    while curNum in pathDict.keys():
        path.appendleft(curNum)
        curNum = pathDict[curNum]

    # 最终耗费,路径长度,路径
    return nid_gDict[endNum], len(path), path



# 启发函数
def heuristic(num, parent_num):
    cnt=0
    for i in range(9):
        if not getDig(num,i)==getDig(endNum,i):
            cnt+=1
    return cnt



# nid 为 整型 Num
# (另一种方式是 字符串
def aStar(startNid: '起点',
          endNid: '终点集',
          getNeighbour: '搜索子节点生成规则',
          heuristic: '启发函数',
          buildRes: '路径和结果细则',
          printPath=True,
          ) -> (int, int, deque):

    # 记录耗费
    nid_gDict = dict()

    # 记录路径
    pathDict=dict()

    # 封锁集合 f
    closeDict = dict()

    # 开放集合
    openDict = dict()

    # 探测堆
    # 堆顶节点总是f值最小的点
    detH = []






    # A*算法主体
    def aStarProc(printPath):

        # 起点
        heapq.heappush(detH, (0, startNid))
        nid_gDict[startNid] = 0

        openDict[startNid] = 0

        while detH:
            # 取f值最小的节点
            cur = heapq.heappop(detH)

            # 获取当前节点信息
            curf = cur[0]
            curNid = cur[1]
            curCost = nid_gDict[curNid]


            # 以下语句用于排除 被丢弃的状态节点
            # 这些被丢弃的状态节点 在访问到它们之前,已经有更好的值了
            if curNid in closeDict:
                continue

            # 开放集与封锁集的调整
            openDict.pop(curNid)
            closeDict[curNid] = curf

            # 若到达终点
            if curNid == endNid:
                # 构建路径和最终地图
                return buildRes(printPath,
                                pathDict=pathDict,
                                openDict=openDict,
                                closeDict=closeDict,
                                nid_gDict=nid_gDict,
                                )


            # 拓展子节点
            for mi,neNid in enumerate(getNeighbour(curNid)):


                # 移动耗费g
                newcost=curCost+ 1


                # A*函数
                newNef=newcost+heuristic(neNid,curNid)


                # 发现了更好的f值,重新取出节点
                if neNid in closeDict and newNef<closeDict[neNid]:
                    closeDict.pop(neNid)
                if neNid in openDict and newNef<openDict[neNid]:
                    openDict.pop(neNid)

                # 进入探测子节点
                if neNid not in openDict and neNid not in closeDict:
                    # 修改记录的耗费值
                    nid_gDict[neNid]=newcost

                    # 插入到开放集
                    heapq.heappush(detH, (newNef, neNid))
                    openDict[neNid] = newNef

                    # 打印
                    print()
                    print(show8dig(curNid))
                    print('|')
                    print( show8dig(neNid))
                    print()


                    # 路径保存
                    pathDict[neNid]=curNid

        print('no path')
        return None

    return aStarProc(printPath)















if __name__ == '__main__':
    print(aStar(startNid=startNum,
                endNid=endNum,
                getNeighbour=getNeighbour,
                heuristic=heuristic,
                buildRes=buildRes,
                printPath=True))


