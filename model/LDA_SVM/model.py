import os
import time
import pickle
import gensim
import numpy as np
from sklearn.svm import LinearSVC
from gensim.models.callbacks import PerplexityMetric
from gensim.matutils import corpus2dense

classes = ['travel', 'news', 'business', 'house', 'it',
           'career', 'mil', 'sports', '2008', 'auto',
           'health', 'women', 'cul', 'yule', 'learning']


def construct_corpus(train_path='../../data/train.txt',
                     corpus_path='./tmp/corpus.pkl'):
    t1 = time.time()
    print('获取标签和数据...')
    train_labels, docs = [], []
    with open(train_path, 'r', encoding='utf8') as f:
        for line in f:
            lss = line.strip().split(' ', 1)
            if len(lss) < 2:
                continue
            label, doc = lss
            train_labels.append(classes.index(label))
            docs.append(doc.split(' '))
    t2 = time.time()
    print("获取标签和数据用时:{}秒".format(t2 - t1))

    print('构建词典...')
    dic = gensim.corpora.Dictionary(docs)
    t3 = time.time()
    print("构建词典用时:{}秒".format(t3 - t2))

    print('移除只出现一次的词...')
    # 移除文本中只出现一次的词
    once_ids = [tokenid for tokenid, wordfreq in dic.dfs.items() if wordfreq < 2]
    dic.filter_tokens(once_ids)
    dic.compactify()
    t4 = time.time()
    print("移除会出现一次的词用时:{}秒".format(t4 - t3))

    print('构建语料...')
    train_corpus = [dic.doc2bow(doc) for doc in docs]
    t5 = time.time()
    print("构建语料用时:{}秒".format(t5 - t4))

    print('持久化...')
    with open(corpus_path, 'wb') as f:
        pickle.dump((dic, train_corpus, train_labels), f)
    t6 = time.time()
    print("构建语料用时:{}秒".format(t6 - t5))

    t7 = time.time()
    print('持久化词典，语料和标签。共用时：{}秒。'.format(t7 - t1))


def select_topicnum(corpus_path):
    """
    目的:
        计算选择不同主题个数时的困惑度，然后选择困惑度最小的主题个数
    输出结果:
        模型主题个数：10；困惑度：632.8538665049322 生成模型用时：883.4571077823639s
        模型主题个数：11；困惑度：708.9936581989566 生成模型用时：896.4127230644226s
        模型主题个数：12；困惑度：759.691150101568 生成模型用时：904.9241600036621s
        模型主题个数：13；困惑度：812.2574003901187 生成模型用时：960.4454367160797s
        模型主题个数：14；困惑度：851.1403982394958 生成模型用时：950.9891333580017s
        模型主题个数：15；困惑度：883.9923012414139 生成模型用时：940.971542596817s
        模型主题个数：16；困惑度：934.0987491348716 生成模型用时：943.1974310874939s
        模型主题个数：17；困惑度：992.8325527591637 生成模型用时：988.3116827011108s
        模型主题个数：18；困惑度：1039.1886426842127 生成模型用时：990.8557188510895s
        模型主题个数：19；困惑度：1092.1486445253286 生成模型用时：1011.110119342804s
        模型主题个数：20；困惑度：1145.3192878691325 生成模型用时：1003.9530656337738s
        计算并持久化主题模型用时：10488.662911891937秒
    """
    t1 = time.time()
    # 加载词典, 语料
    with open(corpus_path, 'rb') as f:
        dic, train_corpus, _ = pickle.load(f)
    for topic_num in range(10, 21):
        t21 = time.time()
        print("生成模型...")
        model = gensim.models.LdaModel(train_corpus, id2word=dic, num_topics=topic_num)
        # 计算困惑度
        pp = PerplexityMetric(train_corpus).get_value(model=model)
        print("模型主题个数：{}；困惑度：{}".format(topic_num, pp))
        t22 = time.time()
        print("生成模型用时：{}s".format(t22 - t21))
    t2 = time.time()
    print("计算并持久化主题模型用时：{}秒".format(t2 - t1))


def get_ldamodel(corpus_path, model_dir, best_topicnum=10):
    """
    目的：
        计算主题并保存
    运行：
        加载词典、语料和计算主题模型共用时：466.43783164024353s
        持久化模型:./ model / model_15.pkl用时：0.769057035446167s
        持久化主题向量:./ model / matrix_15.pkl用时：249.6560516357422s
        共用时：716.8629403114319s
    """
    t1 = time.time()
    # 加载词典, 语料
    print('加载词典和语料...')
    with open(corpus_path, 'rb') as f:
        dic, train_corpus, train_labels = pickle.load(f)
    model = gensim.models.LdaModel(train_corpus, id2word=dic, num_topics=best_topicnum)
    t2 = time.time()
    print("加载词典、语料和计算主题模型共用时：{}s".format(t2 - t1))

    print("持久化模型和主题向量...")
    path = model_dir + 'model_' + str(best_topicnum) + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    t3 = time.time()
    print("持久化模型:" + path + "用时：{}s".format(t3 - t2))

    matrix = corpus2dense(model[train_corpus], num_terms=best_topicnum)
    path = model_dir + 'matrix_' + str(best_topicnum) + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump((matrix.T, np.array(train_labels)), f)
    t4 = time.time()
    print("持久化主题向量:" + path + "用时：{}s".format(t4 - t3))

    print("共用时：{}s".format(t4 - t1))


def consturt_test(model_dir, test_path, corpus_path, feature_test_path, best_topicnum):
    """构造测试集"""
    # 加载lda模型
    print('加载lda模型...')
    filenames = os.listdir(model_dir)
    modelname = [item for item in filenames if 'model' in item][0]
    modelpath = model_dir + modelname
    t1 = time.time()
    with open(modelpath, 'rb') as f:
        ldamodel = pickle.load(f)
    t2 = time.time()
    print('加载lda模型用时：{}s'.format(t2 - t1))

    # 加载测试集
    print('加载测试集...')
    test_labels, docs = [], []
    with open(test_path, 'r', encoding='utf8') as f:
        for line in f:
            lss = line.strip().split(' ', 1)
            if len(lss) < 2:
                continue
            label, doc = lss
            test_labels.append(classes.index(label))
            docs.append(doc.split(' '))
    t3 = time.time()
    print('加载测试集用时：{}s'.format(t3 - t2))

    # 加载词典
    print('加载词典...')
    with open(corpus_path, 'rb') as f:
        dic, _, _ = pickle.load(f)
    t4 = time.time()
    print('加载词典用时：{}s'.format(t4 - t3))

    # 构建测试集特征
    print('构建测试集特征...')
    test_corpus = [dic.doc2bow(doc) for doc in docs]
    X_test = corpus2dense(ldamodel[test_corpus], num_terms=best_topicnum).T
    y_test = np.array(test_labels)
    t5 = time.time()
    print('构造测试集特征用时：{}s'.format(t5 - t4))

    print('持久化测试集特征...')
    with open(feature_test_path, 'wb') as f:
        pickle.dump((X_test, y_test), f)
    t6 = time.time()
    print('持久化测试集特征：{}s'.format(t6 - t5))
    print('构建测试集共用时：{}s'.format(t6 - t1))


def svm_classifer(model_dir, feature_test_path):
    # 加载模型和主题向量
    t1 = time.time()
    filenames = os.listdir(model_dir)
    matrixname = [item for item in filenames if 'matrix' in item][0]
    matrixpath = model_dir + matrixname

    """加载训练集和测试集"""
    print('加载训练集和测试集...')
    with open(matrixpath, 'rb') as f:
        X_train, y_train = pickle.load(f)
    with open(feature_test_path, 'rb') as f:
         X_test, y_test = pickle.load(f)
    t2 = time.time()
    print('构造测试集用时：{}s'.format(t2 - t1))

    print('训练svm模型...')
    svm = LinearSVC(class_weight='balanced', dual=False)
    svm.fit(X_train, y_train)
    t3 = time.time()
    print('训练模型用时：{}s'.format(t3 - t2))
    y_pred = svm.predict(X_test)
    print("测试集正确率:", sum(y_test == y_pred) / y_test.shape[0])
    t4 = time.time()
    print('训练svm和测试共用时：{}s'.format(t4 - t1))


if __name__ == '__main__':
    train_path = '../../data/train.txt'
    test_path = '../../data/test.txt'
    corpus_path = './tmp/corpus.pkl'
    feature_test_path = './tmp/feature_test.pkl'
    model_dir = './model/'

    # step1: 持久化词典，词袋模型，标签； 大约运行20min
    construct_corpus(train_path, corpus_path)

    # step2: 选择lda主题模型主题个数(可以不运行, 用来判断选多少个主题)
    # select_topicnum(corpus_path)

    # step3: 根据主题个数计算主题模型
    get_ldamodel(corpus_path, model_dir, best_topicnum=200)

    # step4: 构造测试集
    consturt_test(model_dir, test_path, corpus_path, feature_test_path=feature_test_path, best_topicnum=200)

    # step5: SVM分类
    svm_classifer(model_dir, feature_test_path)
    """
        加载词典和语料...
        加载词典、语料和计算主题模型共用时：3878.746595144272s
        持久化模型和主题向量...
        持久化模型:./ model / model_200.pkl用时：5.131778240203857s
        持久化主题向量:./ model / matrix_200.pkl用时：827.6181101799011s
        共用时：4711.496483564377s
        加载lda模型...
        加载lda模型用时：5.302685737609863s
        加载测试集...
        加载测试集用时：13.917525291442871s
        加载词典...
        加载词典用时：513.6257853507996s
        构建测试集特征...
        构造测试集特征用时：624.315593957901s
        持久化测试集特征...
        持久化测试集特征：1.7989692687988281s
        构建测试集共用时：1158.9605596065521s
        加载训练集和测试集...
        构造测试集用时：1.8990120887756348s
        训练svm模型...
        训练模型用时：78.86468720436096s
        测试集正确率: 0.8017792025168625
        训练svm和测试共用时：82.10292387008667s
    """
