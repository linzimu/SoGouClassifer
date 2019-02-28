import xml.etree.cElementTree as ET
import jieba
import re
import os


def getcontent(text):
    """抽取指定text需要的内容"""
    root = ET.fromstring(text)
    record_url = root.find('url').text
    record_text = root.find('content').text
    if not record_text or not record_url:
        return None, None
    else:
        record_class = re.findall(r'http://(\w+)\.', record_url)[0]
        record_text = ' '.join(jieba.cut(record_text))
        print(record_class, record_text)
        return record_class, record_text


def save_records(filepath='../data/news_sohusite_xml.dat'):
    """抽取文件中需要的内容并保存到新的文件中"""
    with open(filepath, encoding='gb18030') as f:
        res = ''
        path, filename = filepath.rsplit('\\', 1)
        filename = '.'.join(filename.split('.')[-2:])
        fw = open(path + '/new_data/' + filename, 'w', encoding='utf8')
        for i, line in enumerate(f, 1):
            if i % 6 == 1 and res:
                record_class, record_text = getcontent(res)
                if record_class and record_text:
                    fw.write(record_class + '\t' + record_text + '\n')
                res = line
                # break
            elif i % 6 == 2:
                res += line.replace('&', '')
            else:
                res += line
        fw.close()


def get_all(path='../data/SogouCS'):
    """抽取指定目录下的所有文件中的指定内容到新文件中"""
    filenames = os.listdir(path)
    for filename in filenames:
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            print(filepath)
            save_records(filepath)


def merge_files(path='../data/SogouCS/new_data', stop_file='../data/stop_words.txt'):
    """合并文件并去除文件中的停用词"""
    stopwords = []
    with open(stop_file, 'r', encoding='gb18030') as f:
        for line in f:
            stopwords.append(line.strip())
    filenames = os.listdir(path)
    fw = open('../data/all_data.txt', 'w', encoding='utf8')
    for i, filename in enumerate(filenames, 1):
        filepath = os.path.join(path, filename)
        print(filepath)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf8') as f:
                for line in f:
                    tmp = [item for item in line.strip().split() if item not in stopwords]
                    fw.write(' '.join(tmp) + '\n')
        # if i == 1:
        #     break
    fw.close()
    print('文件合并完成！')


def file_stat(path='../data/all_data.txt'):
    with open(path, 'r', encoding='utf8') as f:
        file_classes = set()
        for i, line in enumerate(f, 1):
            file_classes.add(line.split('\t')[0])
        print(i, file_classes)
        # 419595 {'travel', 'news', 'business', 'house', 'it', 'career', 'mil', 'sports', '2008', 'auto', 'health', 'women', 'cul', 'yule', 'learning'}


if __name__ == '__main__':
    # step1: 抽取原始数据中需要的内容到单独的文件中
    # get_all()
    # step2: 合并包含需要的内容到一个文件中
    merge_files()
    # step3: 统计相关特征
    # file_stat()

    pass
