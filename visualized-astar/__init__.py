from collections import deque
from functools import reduce
from queue import Queue

import heapq
import numpy as np
from numpy import *
import matplotlib.pyplot as plt



# 强制numpy打印所有数据
set_printoptions(threshold=np.NaN)

# 颜色字典
colorList = ['green', 'white', 'red', 'purple','blue','yellow']
rgbList = [(46, 255, 85), (255, 255, 255), (205, 92, 92), (139, 101, 8),(50,50,234),(234,241,50)]
colorDict = {color: RGB for color, RGB in zip(colorList, rgbList)}

# 地图尺寸
gridSize = (35, 35)

# 地图颜色映射 14*14
gridCorMap = np.zeros(gridSize + (3,))

# 起始点
startPoint = (18,34)
# 终点
endPoint = (10,4)

# 阻塞地块 set
blockedSet = {(9, 6), (9, 7), (9, 8), (9, 9), (8, 9), (3, 2), (3, 3), (3, 4), (2, 4), (1, 4), (9, 11), (8, 11),
              (7, 9),(10,9),(11,9),(12,9),(13,9),(9,0),(9,1),(9,2) ,(9,5),(9,6),(11,6),(12,6),(13,6),(6,9),
              (5,9),(18,32),(19,32),(20,32),(21,32),(17,32),(17,33),(21,33),(17,34),(13,5),(14,5),(15,5),(16,5),(17,5),(18,5),(19,5),(20,5),(10,6)}




# ax　坐标系




# 绘图
def drawColor(point, color):
    gridCorMap[point[0], point[1], :] = tuple(RGB / 255 for RGB in color)




# 欧几里得距离
def euclidDist(point):
    return ((point[0] - endPoint[0]) ** 2 + (point[1] - endPoint[1]) ** 2) ** 0.5

# 曼哈顿距离
def manhattanDist(point):
    return abs(point[0] - endPoint[0]) + abs(point[1] - endPoint[1])

# 切比雪夫距离
def chebyshevDist(point):
    return max(abs(point[0] - endPoint[0]),abs(point[1] - endPoint[1]))

def aStar(printPath=True):


    # 移动向量
    move_v = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1))

    # 记录耗费
    point_gDict = dict()

    # 记录路径
    pathDict=dict()

    # 封锁集合 f
    closeDict = dict()

    # 开放集合
    openDict = dict()

    # 探测堆
    # 堆顶节点总是f值最小的点
    detH = []


    # 耗散函数
    def g(point):
        return point_gDict[point]


    # 启发函数
    # http://blog.jobbole.com/84694/
    # 若h(node)<=h*(node),且起点与终点之间存在通路,则A*算法能返回最短路径
    def heuristic(point,parent_point):

        # 距离启发参数
        distH=euclidDist(point)

        # 方向启发参数
        # 叉乘积　(a0 * b1)-(b0 * a1)
        va=list(map(lambda e1,e2:e1-e2,point,endPoint))
        vb=list(map(lambda e1,e2:e1-e2,parent_point,endPoint))
        directH=abs(va[0]*vb[1] -vb[0]* va[1] )



        # 因子
        return distH*1.4+directH*0.007



    # 构造最佳路径
    # minF/maxF 探测过程中出现过的非零最高/最低f值,用于绘图
    def buildRes(printPath,minF,maxF):


        #  阻塞地块 绘制
        for bsi in blockedSet:
            drawColor(bsi, colorDict['red'])

        # 对每一个曾经探测到的节点绘制评估值
        for pDict in (openDict,closeDict):
            for point, f in pDict.items():
                # 相对f值 (起点f值为0)
                relaF=(f-minF)/(maxF-minF)

                # 绘制f值 越白代表f值越小,代表评估意义越高
                pColor=tuple(rgbi - 255 * relaF for rgbi in colorDict['white'])
                drawColor(point, pColor)

        # 路径构造和绘制
        path = deque()
        curPoint = endPoint
        while curPoint in pathDict.keys():
            path.appendleft(curPoint)
            curPoint = pathDict[curPoint]

            # 打印路径判断
            if printPath:
                drawColor(curPoint, colorDict['blue'])

        # 起点绘制
        drawColor(startPoint, colorDict['yellow'])
        # 终点绘制
        drawColor(endPoint, colorDict['green'])


        # 最终耗费,路径长度,路径
        return g(endPoint),len(path),path




    # 不越界判断
    def isInBounds(point):
        for i in range(len(gridSize)):
            if point[i] < 0 or point[i] >= gridSize[i]:
                return False
        return True

    # 障碍判断
    def isOnBlocked(point):
        return point in blockedSet

    # 返回上下左右的位置
    def getNeighbour(point):
        return [tuple(point[i] + move_vj[i] for i in range(len(point))) for move_vj in move_v]




    # A*算法主体
    def aStarProc(printPath):
        # minF/maxF 探测过程中出现过的非零最高/最低f值,用于绘图
        minF=float("inf")
        maxF=0

        # 起点
        heapq.heappush(detH, (0, startPoint))
        point_gDict[startPoint] = 0
        openDict[startPoint] = 0

        while detH:
            # 取f值最小的节点
            cur = heapq.heappop(detH)

            # 获取当前节点信息
            curf = cur[0]
            curPoint = cur[1]
            curCost = point_gDict[curPoint]


            # 以下语句用于排除 被丢弃的状态节点
            # 这些被丢弃的状态节点 在访问到它们之前,已经有更好的值了
            if curPoint in closeDict:
                continue

            # 开放集与封锁集的调整
            openDict.pop(curPoint)
            closeDict[curPoint] = curf

            # 若到达终点
            if curPoint == endPoint:
                # 构建路径和最终地图
                return buildRes(printPath,minF=minF,maxF=maxF)


            # 拓展子节点
            for mi,nePoint in enumerate(getNeighbour(curPoint)):
                # 准入判断
                judgeBox = isInBounds(nePoint),\
                           not isOnBlocked(nePoint)
                if judgeBox == (1, 1):
                    # 移动耗费
                    newcost=(curCost+ 1) if mi < 4 else (curCost + 2**0.5)


                    # A*函数
                    newNef=newcost+heuristic(nePoint,curPoint)


                    # 更新非零最大/最小f值
                    maxF=max(maxF,newNef)
                    if newNef>0:
                        minF=min(minF,newNef)

                    # 发现了更好的f值,重新取出节点
                    if nePoint in closeDict and newNef<closeDict[nePoint]:
                        closeDict.pop(nePoint)
                    if nePoint in openDict and newNef<openDict[nePoint]:
                        openDict.pop(nePoint)

                    # 进入探测子节点
                    if nePoint not in openDict and nePoint not in closeDict:
                        # 修改记录的耗费值
                        point_gDict[nePoint]=newcost



                        # 插入到开放集
                        heapq.heappush(detH, (newNef, nePoint))
                        openDict[nePoint] = newNef

                        # 路径保存
                        pathDict[nePoint]=curPoint

        print('no path')
        return None

    return aStarProc(printPath)

# 可视化
def visualize():

    # 新建图
    fig = plt.figure()
    ax = fig.gca()

    # 标签
    ax.set_xticks(np.arange(0, gridSize[0], 1))
    ax.set_yticks(np.arange(0, gridSize[1], 1))

    # 热点图,用来做方块网格
    plt.imshow(gridCorMap, interpolation='nearest')

    # 网格设置 ,minor 小调 钢琴黑键
    plt.grid(True, linestyle='-', color="black", linewidth="2", which='minor')

    # 棒棒
    plt.colorbar()

    # 显示图
    plt.show()


if __name__ == '__main__':
    bestCost,length,path=aStar(printPath=True)
    print('最低耗费:',bestCost)
    print('路径长度:',length)
    print('路径:',path)
    visualize()
