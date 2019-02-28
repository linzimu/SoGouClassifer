import collections
import math
import random
import time

# 总共有15种新闻类别，我们给每个类别一个编号
labels = ['travel', 'news', 'business', 'house', 'it',
          'career', 'mil', 'sports', '2008', 'auto',
          'health', 'women', 'cul', 'yule', 'learning']


def shuffle(inFile, trainText, testText):
    """
        简单的乱序操作，用于生成训练集和测试集
    """
    textLines = []
    with open(inFile, 'r', encoding='utf8') as f:
        for line in f:
            textLines.append(line.strip())

    print("正在准备训练和测试数据，请稍后...")
    random.shuffle(textLines)
    num = len(textLines)
    print('文章总数：', num)
    train = textLines[:3 * (num // 5)]
    test = textLines[3 * (num // 5):]
    print("准备训练和测试数据准备完毕，下一步...")
    with open(trainText, 'w', encoding='utf8') as f:
        for item in train:
            f.write(item + '\n')
    with open(testText, 'w', encoding='utf8') as f:
        for item in test:
            f.write(item + '\n')


def label2id(label):
    return labels.index(label)


def doc_dict():
    """
        构造和类别数等长的0向量
    """
    return [0] * len(labels)


def mutual_info(N, Nij, Ni_, N_j):
    return Nij / N * math.log2((N * Nij + 1) / (Ni_ * N_j))


def count_for_cates(trainFile, featureFile):
    """
        遍历文件，统计每个词在每个类别出现的次数，和每类包含的单词数
        并写入结果特征文件
    """
    docCount = [0] * len(labels)  # 记录各个类别分别包含多少单词(包括重复单词)
    wordCount = collections.defaultdict(doc_dict)  # 记录每个单词每个类别出现的频次

    trainText = []
    with open(trainFile, 'r', encoding='utf8') as f:
        for line in f:
            trainText.append(line.strip())

    # 扫描文件和计数
    for line in trainText:
        lss = line.strip().split(' ', 1)
        if len(lss) < 2:
            continue
        label, text = lss
        index = label2id(label)
        words = text.split(' ')
        docCount[index] += len(words)
        for word in words:
            wordCount[word][index] += 1

    # 计算互信息值
    print("计算互信息，提取关键/特征词中，请稍后...")
    miDict = collections.defaultdict(doc_dict)  # 记录每个单词和每个类别之间的互信息
    N = sum(docCount)
    print('文档中单词总数:', N)
    # 计算单词和类别的互信息
    for k, vs in wordCount.items():
        for i in range(len(vs)):
            N11 = vs[i]  # 包含特征词k,同时属于第i类的文档数
            N10 = sum(vs) - N11  # 包含特征词k，同时不属于第i类的文档数
            N01 = docCount[i] - N11  # 不包含特征词k，同时属于第i类的文档数
            N00 = N - N11 - N10 - N01  # 不包含特征词k，同时不属于第i类的文档数
            mi = mutual_info(N, N11, N10 + N11, N01 + N11) \
                 + mutual_info(N, N01, N00 + N01, N01 + N11) \
                 + mutual_info(N, N10, N10 + N11, N00 + N10) \
                 + mutual_info(N, N00, N00 + N01, N00 + N10)
            miDict[k][i] = mi

    fWords = set()  # 这里注意fWords是集合，会去除重复的关键词
    for i in range(len(docCount)):
        # 第i个类别与单词的互信息从大到小排序
        sortedDict = sorted(miDict.items(), key=lambda x: x[1][i], reverse=True)
        # 选出第i个类别与单词的互信息最大的前100个单词
        for j in range(100):
            fWords.add(sortedDict[j][0])
    with open(featureFile, 'w', encoding='utf8') as f:
        f.write(str(docCount) + "\n")
        for fword in fWords:
            f.write(fword + "\n")
    print("特征词写入完毕...")


def load_feature_words(featureFile):
    """
        从特征文件导入特征词
    """
    with open(featureFile, 'r', encoding='utf8') as f:
        # 各个类分别包含多少个单词
        docCounts = eval(f.readline())
        features = set()
        # 读取特征词
        for line in f:
            features.add(line.strip())
    return docCounts, features


def train_bayes(featureFile, trainFile, modelFile):
    """
        训练贝叶斯模型，实际上计算每个类中特征词的出现次数
    """
    print("使用朴素贝叶斯训练中...")

    docCounts, features = load_feature_words(featureFile)
    wordCount = collections.defaultdict(doc_dict)  # 每个特征词每个类别出现的频次
    tCount = [0] * len(docCounts)  # 各个类别分别包含多少个特征词
    for line in open(trainFile, 'r', encoding='utf8'):
        lss = line.strip().split(' ', 1)
        if len(lss) < 2:
            continue
        label, text = lss
        index = label2id(label)
        words = text.split(' ')
        for word in words:
            if word in features:
                tCount[index] += 1
                wordCount[word][index] += 1
    # 拉普拉斯平滑
    print("训练完毕，写入模型...")
    with open(modelFile, 'w', encoding='utf8') as f:
        for k, v in wordCount.items():
            # 求出各个类别下，各个特征词的频率（加入拉普拉斯平滑）
            scores = [(v[i] + 1) / (tCount[i] + len(wordCount)) for i in range(len(v))]
            f.write(k + " " + str(scores) + "\n")


def load_model(modelFile):
    """
        从模型文件中导入计算好的贝叶斯模型
    """
    print("加载模型中...")
    f = open(modelFile, 'r', encoding='utf8')
    scores = {}
    for line in f:
        lss = line.strip().split(' ', 1)
        if len(lss) < 2:
            continue
        word, counts = lss
        scores[word] = eval(counts)
    f.close()
    return scores


def predict(featureFile, modelFile, testFile):
    """
        预测文档的类标，标准输入每一行为一个文档
    """
    docCounts, features = load_feature_words(featureFile)
    docScores = [math.log(count * 1.0 / sum(docCounts)) for count in docCounts]  # 各个类别包含单词的比例(取对数)
    scores = load_model(modelFile)
    rCount = 0
    docCount = 0
    print("正在使用测试数据验证模型效果...")

    testText = []
    with open(testFile, 'r', encoding='utf8') as f:
        for i, line in enumerate(f, 1):
            testText.append(line.strip())

    for line in testText:
        lss = line.strip().split(' ', 1)
        if len(lss) < 2:
            continue
        label, text = lss
        index = label2id(label)
        words = text.split(' ')
        preValues = list(docScores)
        for word in words:
            if word in features:
                for i in range(len(preValues)):
                    preValues[i] += math.log(scores[word][i])
        m = max(preValues)
        res = preValues.index(m)
        if res == index:
            rCount += 1
        docCount += 1
    print("总共测试文本量: %d , 预测正确的类别量: %d, 朴素贝叶斯分类器准确度:%f" % (docCount, rCount, rCount * 1.0 / docCount))


if __name__ == "__main__":
    t1 = time.time()
    # 相关文件路径
    inFile = '../../data/all_data.txt'
    trainFile = '../../data/train.txt'
    testFile = '../../data/test.txt'
    featureFile = './tmp/feature_file.model'
    modelFile = './tmp/model_file.model'

    # step1:划分训练集和测试集
    shuffle(inFile, trainFile, testFile)
    t2 = time.time()
    print('划分数据集用时：{}秒\n'.format(t2 - t1))

    # step2:提取特征
    count_for_cates(trainFile, featureFile)
    t3 = time.time()
    print('提出特征用时：{}秒\n'.format(t3 - t2))

    # step3:训练模型
    train_bayes(featureFile, trainFile, modelFile)
    t4 = time.time()
    print('训练模型用时：{}秒\n'.format(t4 - t3))

    # step4:预测测试集
    predict(featureFile, modelFile, testFile)
    t5 = time.time()
    print('测试模型用时：{}秒'.format(t5 - t4))
    print('总用时：{}秒'.format(t5 - t1))
    # 1. 不去停用词，使用MI提取特征，在利用NB分类准确率为0.754769
    # 2. 去停用词，使用MI提取特征，在利用NB分类准确率为0.786056
