import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    '''
    打开文本文件textSet.txt并逐行读取
    :return:
    '''
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # x1，x2
        labelMat.append(int(lineArr[2]))  # 类别标签
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''
    通过梯度上升算法找到最佳回归系数，拟合出Logistic回归模型的最佳参数
    :param dataMatIn: 二维numpy数组，每列分别代表每个不同的特征，每行则代表每个训练样本
    :param classLabels:
    :return:
    '''
    # 转换成numpy矩阵数据类型
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()  # 为了便于矩阵运算，将行向量转换成列向量
    m, n = np.shape(dataMatrix)
    alpha = 0.001  # 目标移动的步长
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1))  # 构造全一矩阵（n行1列）
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 列向量，元素个数等于样本个数
        error = (labelMat - h)  # 计算差值
        weights = weights + alpha * dataMatrix.transpose() * error  # 按照差值的方向调整回归系数
    return weights

def plotBestFit(weights):
    '''
    画出数据集和Logistic回归最佳拟合直线
    :param weights:
    :return:
    '''
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    '''
    随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :return:
    '''
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # h和error全是数值
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    '''
    改进的随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :return:
    '''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 每次迭代都调整
            randIndex = int(np.random.uniform(0, len(dataIndex)))  # 随机选取更新
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    '''
    以回归系数和特征向量作为输入计算对应的sigmoid值
    :param inX:
    :param weights:
    :return:
    '''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeight = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainingWeight)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


def main():

    '''
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    plotBestFit(weights.getA())  # 将weight矩阵转换成数组
    '''
    '''
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent0(np.array(dataArr), labelMat)
    plotBestFit(weights)
    '''
    print(multiTest())




if __name__ == '__main__':
    main()