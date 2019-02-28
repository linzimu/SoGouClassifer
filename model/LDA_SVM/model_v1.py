import os
import time
import pickle
import gensim
import numpy as np
from sklearn.svm import LinearSVC
from gensim.models.callbacks import PerplexityMetric

classes = ['travel', 'news', 'business', 'house', 'it',
           'career', 'mil', 'sports', '2008', 'auto',
           'health', 'women', 'cul', 'yule', 'learning']


def construct_corpus(data_path='../../data/train.txt',
                     dic_path='./tmp/corpus.dic',
                     corpus_path='./tmp/corpus.pkl',
                     label_path='./tmp/labels.pkl'):
    t1 = time.time()
    print('获取标签和数据...')
    labels, docs = [], []
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f:
            lss = line.strip().split(' ', 1)
            if len(lss) < 2:
                continue
            label, doc = lss
            labels.append(classes.index(label))
            docs.append(doc.split(' '))

    print('构建词典...')
    dic = gensim.corpora.Dictionary(docs)
    print('移除只出现一次的词...')
    # 移除文本中只出现一次的词
    once_ids = [tokenid for tokenid, wordfreq in dic.dfs.items() if wordfreq < 2]
    dic.filter_tokens(once_ids)
    dic.compactify()

    print('持久化...')
    # step1: 词典持久化
    dic.save(dic_path)
    # step2: 将词袋模型持久化
    corpus_list = [dic.doc2bow(doc) for doc in docs]
    gensim.corpora.MmCorpus.serialize(corpus_path, corpus_list)
    # step3: 将标签持久化
    with open(label_path, 'wb') as f:
        pickle.dump(labels, f)
    t2 = time.time()
    print('持久化词典，词袋模型，标签用时：{}秒'.format(t2 - t1))


def lda_model(dic_path, corpus_path, topic_dir):
    t1 = time.time()
    # 加载词典
    dic = gensim.corpora.Dictionary.load(dic_path)
    # 加载语料
    corpus = gensim.corpora.MmCorpus(corpus_path)
    for topic_num in range(10, 21):
        model = gensim.models.LdaModel(corpus, id2word=dic, num_topics=topic_num)
        # 计算困惑度
        pp = PerplexityMetric(corpus).get_value(model=model)
        print("主题个数：{}；困惑度：{}".format(topic_num, pp))

        topic_path = topic_dir + 'model_' + str(topic_num) + '.pkl'
        print("持久化" + topic_path + "...")
        with open(topic_path, 'wb') as f:
            pickle.dump(model, f)
    t2 = time.time()
    print("计算并持久化主题模型用时：{}秒".format(t2 - t1))


def svm_classifer(corpus_path, dic_path, topic_dir, label_path):
    # 加载词典
    labels, docs = [], []
    with open(corpus_path, 'r', encoding='utf8') as f:
        for line in f:
            lss = line.strip().split(' ', 1)
            if len(lss) < 2:
                continue
            label, doc = lss
            labels.append(classes.index(label))
            docs.append(doc.split(' '))
    dic = gensim.corpora.Dictionary.load(dic_path)
    corpus_list = [dic.doc2bow(doc) for doc in docs]
    with open(label_path, 'rb') as f:
        labels = np.array(pickle.load(f))
    filenames = os.listdir(topic_dir)
    for filename in filenames:
        filepath = os.path.join(topic_dir, filename)
        with open(filepath, 'rb') as f:
            doc_topic_mat = pickle.load(f)
        svm = LinearSVC(class_weight='balanced', dual=False)
        svm.fit(doc_topic_mat, labels)
        y_pred = svm.predict(doc_topic_mat)
        print("训练集正确率:", sum(labels == y_pred) / len(labels))


if __name__ == '__main__':
    data_path = '../../data/train.txt'
    dic_path = './tmp/corpus.dic'
    corpus_path = './tmp/corpus.pkl'
    label_path = './tmp/labels.pkl'
    topic_dir = './model/'
    # step1: 持久化词典，词袋模型，标签
    construct_corpus(data_path, dic_path, corpus_path, label_path)
    # step2: 计算并持久化主题模型
    lda_model(dic_path, corpus_path, topic_dir)
    # step3: SVM分类
    svm_classifer(topic_dir, label_path)
