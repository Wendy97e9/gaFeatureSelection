import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from deap import creator, base, tools, algorithms
import sys

# clf = LogisticRegression()
clf = KNeighborsClassifier(n_neighbors=1) # K-NN
# clf = DecisionTreeClassifier() # CART(决策树)
# clf = SVC(kernel='rbf') # Rbf-svm

def avg(l):
    """
    Returns the average between list elements
    """
    return (sum(l)/float(len(l)))


def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """

    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]
        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)
        # 70%~30%
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42,shuffle=True)
        # X_train_parsed = X_train.drop(X_train.columns[cols], axis=1)
        # X_train_subset = pd.get_dummies(X_train_parsed)
        # X_test_parsed = X_test.drop(X_test.columns[cols], axis=1)
        # X_test_subset = pd.get_dummies(X_test_parsed)
        # clf.fit(X_train_subset, y_train)

        X_train_subset, X_test_subset, y_train, y_test = train_test_split(X_subset, y, test_size=0.30, random_state=42,
                                                                          shuffle=True)
        DR = 1 - (len(X_test_subset.columns) / len(individual))
        clf.fit(X_train_subset, y_train)
        return (clf.score(X_test_subset, y_test),)
        # return (clf.score(X_test_subset, y_test),)

        # 交叉验证


        # apply classification algorithm
        # clf = LogisticRegression()

        # return (avg(cross_val_score(clf, X_subset, y, cv=5)),)
    else:
        return(0,)

#######################################################################################
# 计算数据项i1和i2在特征j上的的差异度
def diff(X, y, i1, i2, j, Xmax, Xmin):
    ddiff = abs(X.iat[i1,j] - X.iat[i2,j])/(Xmax.iat[j] - Xmin.iat[j])
    return ddiff

# 找i的r个同类近邻
def neighbor(df, i, r):
    # neighbor = []
    sameneighbor = df[df["Type"]==df.iat[i,-1]].index.tolist()
    diffneighbor = df[df["Type"]!=df.iat[i,-1]].index.tolist()
    # print("All same class neighbor"+str(sameneighbor))
    rsameneighbor = random.sample(sameneighbor,r)
    # print("r same class neighbor" + str(rsameneighbor))
    rdiffneighbor = random.sample(diffneighbor,r)
    # print("r diff class neighbor" + str(rdiffneighbor))
    return rsameneighbor,rdiffneighbor

# 特征权重评估 X:数据 t:迭代次数 r:随机邻居的数量
def evaluateWeight(df, X, y, t, r, Xmax, Xmin):
    M = len(df.columns)-1   # M是特征数
    N = len(df)
    print(N)
    w = [0.5 for ii in range(M)]
    print(w)
    for ii in range(t):
        i = random.randint(0, N-1)  # 随机一个数据项i
        iirsnb,iirdnb = neighbor(df, i, r)  # 找到同类不同类的R个邻居
        for j in range(M):  # 特征j
            sdiff = 0
            ddiff = 0
            for sc in iirsnb:
                sdiff += diff(X,y,i,sc,j,Xmax,Xmin)
            for dc in iirdnb:
                ddiff += diff(X,y,i,dc,j,Xmax,Xmin)
            w[j] = w[j] - sdiff/M + ddiff/M
        print("权值："+str(w))
    return w

def getWFitness(individual, df, X, y):
    """
    Feature subset fitness function
    """
    # 全部特征集的均值向量
    # xv = X.mean()
    # print(xv)
    # 所有类的均值向量，根据最后一列的值来获取某一类的均值向量
    # xiv = df.groupby(df.columns[-1]).mean()
    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]
        subdf = df.drop(X.columns[cols], axis = 1)  #特征子集
        # print("特征子集的长度："+str(len(subdf.columns)-1))
        # print(individual)
        subX = subdf.iloc[:, :-1]
        subXvec = subX.mean()   #特征子集均值向量
        # print("特征子集均值向量：")
        # print(str(subXvec))
        # print("特征子集分组：")
        grouped = subdf.groupby(subdf.columns[-1])
        # print(grouped.agg("count"))
        n = []  #每一类的样本个数
        C = 0   #分母上的系数，无效
        # gb = groupeddp.get_group("")
        # print(gb)
        for gb in grouped:
            # print("分组")
            n.append(len(gb[1]))
            C += 1/(len(gb[1])-1)
            # print(gb)
        # print("每一组的样本个数："+str(n))
        # print("分母系数："+str(C))
        subXivec = subdf.groupby(subdf.columns[-1]).mean()  #i类特征子集均值向量
        # print("每一类特征子集均值向量：")
        # print(str(subXivec))
        # 计算类间距离
        leijian = subXivec-subXvec
        # print("类间：")
        # print(leijian)
        leijiandistance = 0
        for ii in range(leijian.shape[0]):
            for jj in range(leijian.shape[1]):
                leijiandistance += leijian.iat[ii,jj]*leijian.iat[ii,jj]
        # 类内距离，用分组计算了
        group_index = grouped.size().index.tolist()
        # print("groupindex"+str(group_index))
        group_df = []
        sumleineidistance = 0
        # gb 是索引号，取出对应子数据集，计算类内距离
        i = 0
        for gb in group_index:
            groupi_data = grouped.get_group(gb)
            # print("分组 "+str(gb))
            groupiX = groupi_data.drop(groupi_data.columns[-1],axis = 1) #删除最后一列
            leinei = groupiX-subXivec.loc[gb] #series 和 dataframe运算竟然对了...
            leineidistance = 0
            # print("类内：")
            # print(leinei)
            for ii in range(leinei.shape[0]):
                for jj in range(leinei.shape[1]):
                    leineidistance += leinei.iat[ii, jj] * leinei.iat[ii, jj]
            sumleineidistance += leineidistance*1/(n[i]-1)
            i += 1

            group_df.append((groupi_data))
        # print("类间距离:" + str(leijiandistance))
        # print("类内距离：" + str(sumleineidistance))
        f = leijiandistance/sumleineidistance
        # print("适应度："+str(f))
        return (f,)
    else:
        return(0,)
#######################################################################################
# 交叉
def humming(x,y):
    d = 0
    for i in range(len(x)):
        d += x[i]^y[i]
    # print("海明距离： "+str(d))
    return d

def cxOnePointapt( ind1, ind2):
    '''
    保留高适应度个体的特征，只将低适应度的个体
    '''
    size = min(len(ind1), len(ind2))
    d = humming(ind1, ind2)
    dif = humming(ind1, ind2) / size
    if dif < 0.05:
        return ind1, ind2
    cxpoint = random.randint(1, size - 1)
    # ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    gap = ind1.fitness.values[0] - ind2.fitness.values[0]
    if gap > 0.1:
        if ind1.fitness.values[0] > ind2.fitness.values[0]:
            ind2[cxpoint:] = ind1[cxpoint:]
        else:
            ind1[cxpoint:] = ind2[cxpoint:]
    else:
        ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2

#######################################################################################
def mutFlipBitapt(individual, indpb):
    """Flip the value of the attributes of the input individual and return the
    mutant. The *individual* is expected to be a :term:`sequence` and the values of the
    attributes shall stay valid after the ``not`` operator is called on them.
    The *indpb* argument is the probability of each attribute to be
    flipped. This mutation is usually applied on boolean individuals.

    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be flipped.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i in range(len(individual)):
        rdm = random.random()
        if individual[i] == 1:
            if rdm < 10 * indpb:
                individual[i] = 0
        else:
            if rdm < indpb:
                individual[i] = 1
            # individual[i] = type(individual[i])(not individual[i])

    return individual,
#######################################################################################

def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 单目标，最大值问题
    creator.create("Individual", list, fitness=creator.FitnessMax)  #编码继承List类

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))

    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    # 评价族群
    toolbox.register("evaluate", getFitness, X=X, y=y)
    # toolbox.register("evaluate", getWFitness,df=df, X=X, y=y)
    # 交配 crossover 子代替换父代
    toolbox.register("mate", tools.cxOnePoint)
    # toolbox.register("mate", cxOnePointapt)
    # 变异 0.05 子代替换父代
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # toolbox.register("mutate", mutFlipBitapt, indpb=0.05)
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # 突变
    # 选择
    toolbox.register("select", tools.selTournament, tournsize=3)    # 3选1
    # toolbox.register("select", tools.selRoulette)  # 3选1

    # initialize parameters
    # 生成初始种群
    pop = toolbox.population(n=n_population)
    # print(pop)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm 只取后代构成种群
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)
    # nevals numbers of evaluation
    # return hall of fame
    # 返回最后一代种群，最优解在其中
    return hof


def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    maxAccurcy = 0.0
    # print("bestIndividual Number of HallOfFame:"+str(len(hof)))
    i = 0
    for individual in hof:
        # print(str(i)+str(individual)+str(individual.fitness.values))  # 只有[0]有值
        i += 1
        if(individual.fitness.values[0] > maxAccurcy):
            maxAccurcy = individual.fitness.values[0]
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


def getArguments():
    """
    Get argumments from command-line
    If pass only dataframe path, pop and gen will be default
    """
    dfPath = sys.argv[1]
    if(len(sys.argv) == 4):
        pop = int(sys.argv[2])
        gen = int(sys.argv[3])
    else:
        pop = 10
        gen = 2
    return dfPath, pop, gen


if __name__ == '__main__':
    # get dataframe path, population number and generation number from command-line argument
    dataframePath, n_pop, n_gen = getArguments()
    # read dataframe from csv
    df = pd.read_csv(dataframePath, sep=',')

    N_AF = len(df.iloc[0,:]) - 1

    # encode labels column to numbers
    # y是标签编号
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])

    # 相当于对标签进行离散化
    # X是数据
    X = df.iloc[:, :-1]
    Xmax = X.max()
    Xmin = X.min()


    # get accuracy with all features
    individual = [1 for i in range(len(X.columns))]
    print("Accuracy with all features: \t" + str(getFitness(individual, X, y)) + "\n")
    # print("Accuracy with all features: weighted\t" + str(getWFitness(individual, df, X, y)) + "\n")

    # apply genetic algorithm
    hof = geneticAlgorithm(X, y, n_pop, n_gen)
    # select the best individual
    accuracy, individual, header = bestIndividual(hof, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

    N_SF = len(header)

    print('\n\ncreating a new classifier with the result')

    # read dataframe from csv one more time
    df = pd.read_csv(dataframePath, sep=',')

    # with feature subset
    X = df[header]


    # 逻辑回归函数
    # clf = LogisticRegression() # clf 改成了全局变量

    # 70%-30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)
    clf.fit(X_train, y_train)
    scores = clf.score(X_test, y_test)
    print("Accuracy with Feature Subset: \t" + str(scores) + "\t Decrease Rate:\t" + str(1 - N_SF / N_AF) + "\n")

    # # 交叉验证评分
    # scores = cross_val_score(clf, X, y, cv=10)
    # 参数依次为 估计方法对象(分类器) 数据特征 数据标签 几折交叉验证
    # print("Accuracy with Feature Subset: \t" + str(avg(scores)) + "\t Decrease Rate:\t" + str(1 - N_SF / N_AF) + "\n")
