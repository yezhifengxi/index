# 代码8-1
import re

# from scipy.misc import imread
import imageio
import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import imageio.v2

# 读取数据
data = pd.read_csv(r'F:\PycharmProjects\1\demo_nlp\utils\第8章\data\message80W.csv', encoding='utf-8', index_col=0, header=None)
data.columns = ['类别', '短信']
data.类别.value_counts()

# 代码8-2
temp = data.短信
temp.isnull().sum()
# 去重
data_dup = temp.drop_duplicates()
# 脱敏
l1 = data_dup.astype('str').apply(lambda x: len(x)).sum()
data_qumin = data_dup.astype('str').apply(lambda x: re.sub('x', '', x))
l2 = data_qumin.astype('str').apply(lambda x: len(x)).sum()
print('减少了' + str(l1 - l2) + '个字符')
# 加载自定义词典
jieba.load_userdict(r'F:\PycharmProjects\1\demo_nlp\utils\第8章\data\newdic1.txt')
# 分词
data_cut = data_qumin.astype('str').apply(lambda x: list(jieba.cut(x)))
# 去停用词
stopword = pd.read_csv(r'F:\PycharmProjects\1\demo_nlp\utils\第8章\data\stopword.txt', sep='ooo', encoding='gbk',
                       header=None, engine='python')
stopword = [' '] + list(stopword[0])
l3 = data_cut.astype('str').apply(lambda x: len(x)).sum()
data_qustop = data_cut.apply(lambda x: [i for i in x if i not in stopword])
l4 = data_qustop.astype('str').apply(lambda x: len(x)).sum()
print('减少了' + str(l3 - l4) + '个字符')
data_qustop = data_qustop.loc[[i for i in data_qustop.index if data_qustop[i] != []]]

# 代码8-3
# 词频统计
lab = [data.loc[i, '类别'] for i in data_qustop.index]
lab1 = pd.Series(lab, index=data_qustop.index)


def cipin(data_qustop, num=10):
    temp = [' '.join(x) for x in data_qustop]
    temp1 = ' '.join(temp)
    temp2 = pd.Series(temp1.split()).value_counts()
    return temp2[temp2 > num]


data_gar = data_qustop.loc[lab1 == 1]
data_nor = data_qustop.loc[lab1 == 0]
data_gar1 = cipin(data_gar, num=5)
data_nor1 = cipin(data_nor, num=30)

# 绘制垃圾短信词云图
back_pic = imageio.imread(r'F:\PycharmProjects\1\demo_nlp\utils\第8章\data\background.jpg')
wc = WordCloud(font_path='C:/Windows/Fonts/simkai.ttf',  # 字体
               background_color='white',  # 背景颜色
               max_words=2000,  # 最大词数
               mask=back_pic,  # 背景图片
               max_font_size=200,  # 字体大小
               random_state=1234)  # 设置多少种随机的配色方案
gar_wordcloud = wc.fit_words(data_gar1)
plt.figure(figsize=(16, 8))
plt.imshow(gar_wordcloud)
plt.axis('off')
plt.savefig(r'F:\PycharmProjects\1\demo_nlp\utils\第8章\tmp\spam.jpg')
plt.show()

# 绘制非垃圾短信词云图
nor_wordcloud = wc.fit_words(data_nor1)
plt.figure(figsize=(16, 8))
plt.imshow(nor_wordcloud)
plt.axis('off')
plt.savefig(r'F:\PycharmProjects\1\demo_nlp\utils\第8章\tmp\non-spam.jpg')
plt.show()

# 代码8-4
num = 10000
adata = data_gar.sample(num, random_state=123)
bdata = data_nor.sample(num, random_state=123)
data_sample = pd.concat([adata, bdata])
cdata = data_sample.apply(lambda x: ' '.join(x))
lab = pd.DataFrame([1] * num + [0] * num, index=cdata.index)
my_data = pd.concat([cdata, lab], axis=1)
my_data.columns = ['message', 'label']


# 代码8-5
# 加载、划分训练集和测试集
def loadDataSet(n, m, my_data):
    data_qs = my_data
    data_qs['num'] = range(len(data_qs))
    data_qustop2 = data_qs[np.array(data_qs.num >= n) & np.array(data_qs.num < m)]
    index = data_qustop2.sample(n=int(len(data_qustop2) * 0.8), random_state=None)
    train_postingList = [data_qustop2.message[i] for i in index.index]
    train_classVec = [data_qustop2.label[i] for i in index.index]
    test_postingList = [data_qustop2.message[i] for i in
                        data_qustop2.index if i not in index.index]
    test_classVec = [data_qustop2.label[i] for i in
                     data_qustop2.index if i not in index.index]
    return train_postingList, train_classVec, test_postingList, test_classVec


# 生成词库
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document.split())
    vocabSet = list(vocabSet)
    return vocabSet


# 词频向量矩阵
def setWordsVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            pass
    return returnVec


# 计算条件概率
def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 样本数量
    numWords = len(trainMatrix[0])  # 特征数量
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 分类为1的概率
    # 初始化分类为1和0的特征
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denmo = 2.0
    p1Denmo = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denmo += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denmo += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denmo)
    p0Vect = np.log(p0Num / p0Denmo)
    return p0Vect, p1Vect, pAbusive


# 判断所属分类
def classifyNB(vec2Classify, p0Vec, p1Vec, PClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(PClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - PClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 代码8-6
def testingNB(n, m, my_data):
    # 加载划分数据集
    train_postingList, train_classVec, test_postingList, test_classVec = loadDataSet(n, m, my_data)
    # 生成词库
    myVocabList = createVocabList(train_postingList)
    # 词频向量矩阵
    trainMat = []
    for postDoc in train_postingList:
        trainMat.append(setWordsVec(myVocabList, postDoc.split()))
    # 计算条件概率
    p0v, p1v, pAb = trainNB(np.array(trainMat), np.array(train_classVec))
    # 测试
    count = 0
    count_11 = 0
    count_00 = 0
    count_10 = 0
    count_01 = 0
    for i in range(len(test_postingList)):
        testEntry = test_postingList[i].split()
        true_class = test_classVec[i]
        # 词频向量
        thisDoc = np.array(setWordsVec(myVocabList, testEntry))
        print(testEntry, '预测为：', classifyNB(thisDoc, p0v, p1v, pAb))
        print('真实类别：%s' % str(true_class))
        pred = classifyNB(thisDoc, p0v, p1v, pAb)
        if str(true_class) == str(pred):
            count += 1
            if str(true_class) == '1' and str(pred) == '1':
                count_11 += 1
            else:
                count_00 += 1
        else:
            if str(true_class) == '1' and str(pred) == '0':
                count_10 += 1
            else:
                count_01 += 1
    print('正确率 %s' % str(int(count) / len(test_postingList)))
    return len(test_postingList), count, count_00, count_01, count_10, count_11


testingNB(0, 20000, my_data)

# 代码8-7
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    my_data.message, my_data.label, test_size=0.2, random_state=123)  # 构建词频向量矩阵
# 训练集
cv = CountVectorizer()  # 将文本中的词语转化为词频矩阵
train_cv = cv.fit_transform(x_train)  # 拟合数据，再将数据转化为标准化格式
train_cv.toarray()
train_cv.shape  # 查看数据大小
cv.vocabulary_  # 查看词库内容
# 测试集
cv1 = CountVectorizer(vocabulary=cv.vocabulary_)
test_cv = cv1.fit_transform(x_test)
test_cv.shape
# 朴素贝叶斯
nb = MultinomialNB()  # 朴素贝叶斯分类器
nb.fit(train_cv, y_train)  # 训练分类器
pre = nb.predict(test_cv)  # 预测

# 代码8-8
# 评价
cm = confusion_matrix(y_test, pre)
cr = classification_report(y_test, pre)
print(cm)
print(cr)
