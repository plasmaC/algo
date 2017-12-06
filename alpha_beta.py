
from collections import OrderedDict

treeDict={}

treeDict[1]=[2,3,4]
treeDict[2]=[5,6]
treeDict[3]=[7,8]
treeDict[4]=[9,10,11]
treeDict[5]=[12]
treeDict[6]=[13,14]
treeDict[7]=[15,16]
treeDict[8]=[17]
treeDict[9]=[18,19]
treeDict[10]=[20,21]
treeDict[11]=[22]
treeDict[12]=[23,24]
treeDict[13]=[25]
treeDict[14]=[26]
treeDict[15]=[27]
treeDict[16]=[28,29]
treeDict[17]=[30]
treeDict[18]=[31]
treeDict[19]=[32]
treeDict[20]=[33]
treeDict[21]=[34]
treeDict[22]=[35,36]
treeDict[23]=[37,38]
treeDict[24]=[39,40,41]
treeDict[25]=[42,43]
treeDict[26]=[44]
treeDict[27]=[45,46]
treeDict[28]=[47]
treeDict[29]=[48]
treeDict[30]=[49]
treeDict[31]=[50,51]
treeDict[32]=[52,53]
treeDict[33]=[54]
treeDict[34]=[55]
treeDict[35]=[56]
treeDict[36]=[57,58]
treeDict[37]=[59,60]
treeDict[38]=[61,62]
treeDict[39]=[63]
treeDict[40]=[64,65,66]
treeDict[41]=[67,68]
treeDict[42]=[69,70]
treeDict[43]=[71,72]
treeDict[44]=[73,74]
treeDict[45]=[75,76]
treeDict[46]=[77,78]
treeDict[47]=[79,80]
treeDict[48]=[81,82]
treeDict[49]=[83]
treeDict[50]=[84]
treeDict[51]=[85,86]
treeDict[52]=[87,88]
treeDict[53]=[89,90]
treeDict[54]=[91,92]
treeDict[55]=[93,94]
treeDict[56]=[95,96]
treeDict[57]=[97,98]
treeDict[58]=[99]



#for i in range(1,100,1):
#    print('treeDict[',i,']=[]',sep='')



leafIDtuple=range(59,100,1)
leafVatuple=[0,5,-3,3,3,-3,0,2,-2,3,5,2,5,-5,0,1,5,1,-3,0,-5,5,-3,3,2,3,-3,0,-1,-2,0,1,4,5,1,-1,-1,3,-3,2,-2]
ValueDict=dict(zip(leafIDtuple,leafVatuple))


#print(len(leafVaList))
#print(ValueDict[71])




# 判断是否终局
def isEnd(node):
    return node in ValueDict.keys()

# 评估终局分数
def evaluate(node):
    return ValueDict[node]


# 可通过比较当前深度奇偶性和根节点奇偶性的异同，判断当前在a层还是b层
# 一般来说根节点的深度为0
# 则a层深度均为偶数
# b层深度均为奇数
def onAlpha(depth):
    return depth%2 == 0


# 生成器
# 由一个节点产生子节点
def getNe(node):
    for ne in treeDict[node]:
        yield ne


# αβ剪枝算法
def alph_bet(root,depth):
    # 保存最佳路径的字典
    pathDict={}

    # αβ主体
    # 第一个参数为节点标号
    # 第二个参数为当前深度
    # 第三个和第四个为 限制范围，用于剪枝
    # 这两个值总是向下传递区间最小的αβ值
    # 也就是说lim_min总是为搜索过程中出现的最大的α值
    #         lim_max总是为搜索过程中出现的最小的β值


    # 该函数返回一个叫cur_value的值
    # cur_value在不同情况下有不同含义
    # 如果当前层是叶子，则cur_value为评估值
    # 如果当前层是a层，则cur_value总是保存当前节点从其子节点接收到的最大值
    # 如果当前层是b层，则cur_value总是保存当前节点从其子节点接收到的最小值
    # 显然当cur_value在lim_min和lim_max之外时，发生剪枝


    # 除此之外，如果cur_value使得当前节点的αβ值区间更小
    # 那么在当前节点扩展新节点时将向下传递新的αβ值 （如何修改则取决于当前节点是哪一层

    # 经过测试和推导，a剪枝只会在b层子节点发生，b剪枝只会在a层子节点发生
    # 所以在a层只需监视cur_value和lim_max
    #     在b层只需监视cur_value和lim_min

    def ab(cur,depth,lim_min,lim_max):
        # 终局
        if isEnd(cur):
            return evaluate(cur)
        # 继续博弈
        else:
            #　生成器
            gen = getNe(cur)
            cutFlag=False

            # 若在a层
            if onAlpha(depth):

                cur_value=float('-inf')


                for ne in gen:
                    if cutFlag:
                        print('结点',ne,'被施行b剪枝',sep='')
                        continue

                    # 递归入口
                    ne_value=ab(ne,depth+1,max(lim_min,cur_value),lim_max)

                    if ne_value > cur_value:
                        cur_value = ne_value
                        pathDict[cur]=ne

                    # b剪枝
                    if cur_value > lim_max:
                        # break
                        cutFlag=True


            # 若在b层
            else:
                cur_value = float('inf')
                for ne in gen:

                    if cutFlag:
                        print('结点', ne, '被施行a剪枝', sep='')
                        continue

                    # 递归入口
                    ne_value=ab(ne,depth+1,lim_min,min(lim_max,cur_value))

                    if ne_value < cur_value:
                        cur_value=ne_value
                        pathDict[cur] = ne

                    # a剪枝
                    if cur_value<lim_min:
                        # break
                        cutFlag=True

            # commit
            return cur_value

    # 路径构造
    def buildPath():
        cur=root
        path=[]
        path.append(root)
        while cur in pathDict.keys():
            path.append(pathDict[cur])
            cur=pathDict[cur]
        return path


    return {'结果':ab(root,depth,float('-inf'),float('inf')),'最佳路径':buildPath()}


if __name__ == '__main__':
    print(alph_bet(1,0))
