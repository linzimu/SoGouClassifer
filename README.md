# SoGouClassifer
搜狗新闻语料分类
----------

## 数据集
数据来源：搜狐新闻数据(SogouCS)——[精简版](https://www.sogou.com/labs/resource/ftp.php?dir=/Data/SogouCS/SogouCS.reduced.tar.gz)(一个月数据, 347MB)；**数据格式**如下：

    <doc>
    <url>页面URL</url>
    <docno>页面ID</docno>
    <contenttitle>页面标题</contenttitle>
    <content>页面内容</content>
    </doc>

## 算法步骤
### step1:
由于，数据是不规范的xml格式构成的，我通过每次读取文件中的6行记录（代表一个文档），再通过正则表达式解析其中的url和content内容。
**在解析`url`时，其中存在`&`无法解析，所以首先替换掉`&`。**最终，得到15个类别的419595条文档记录，并划分**60%训练集**和**40%测试集**。15个类别如下：

	['travel', 'news', 'business', 'house', 'it',
	 'career', 'mil', 'sports', '2008', 'auto',
	 'health', 'women', 'cul', 'yule', 'learning']
### step2:
使用`gensim`工具包。首先，得到原始训练集的词袋模型并将训练集用词袋模型进行表示，**在词袋模型中去除只出现1次的单词**；接着，用LDA模型训练词袋模型表示的训练集对其进行降维处理。最终，得到降维后的特征数据集。
### step3:
测试集处理。首先，用训练集的词袋模型对测试集进行表示。然后，用训练集得到的LDA模型（选取200个主题）对测试集进行同样的降维处理。
### step4:
最后，用SVM对训练集进行训练得到SVM分类器。最终，得到测试集**80.18%的正确率**。SVM训练时，由于`样本数` >> `特征数，`所以`dual`取值`False`，同时`class_weight`取值`'balance'`。
