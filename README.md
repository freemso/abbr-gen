# Chinese Named Entity Generation

## 研究背景

中文实体在很多文本中可能以简称的形式出现。比如「中国农业银行」，可能在文本中直接简称为「农行」。

那么，我们该如何把中文实体全称和它的简称理解为同一个实体？这里可能就会涉及 entity linking 的问题。

换一个思路，如果我们可以对全称自动生成简称，那么这个问题在一定程度上可以被解决。

## 研究问题

输入：中文实体全称；输出：中文实体简称。

例子：「复旦大学」-「复旦」，「北京大学」-「北大」

## 相关工作

这个问题可以形式化为序列标注问题。即，对输入的每个字进行标注 0/1，表示这个字在简称当中是否存在。

## 研究挑战

1. 标注数据的准确率堪忧。我们的标注数据是 entity linking 任务的结果做进一步清洗得到的，其中掺杂了大量错误数据。
2. 评价标准比较模糊。标准答案不是唯一的，比如「中国国际航空公司」，简称是「中国国航」或者「国航」都对。另外，正如上面这个例子所示，简称也是由层级的。但现实是，大部分标注数据对于一个实体只有一个简称。因此，很难对结果做出公允的评判。

## 相关算法

Qian 在 2015 年有一篇论文：

- 使用预训练的一个 `RNN` 模型来判断一个字符序列是否是一个词，起到一个动态词典的作用
- 将 `N-gram` 输入上面的模型得到的输出作为其中一个特征输入到一个 `seq2seq` 的 `RNN` 中
- 用动态词典的目的是为了引入中文词形态学的信息

## 成果

在我们自己的数据集上测试了 Qian 的模型。得到的结果是，RADD 模型的准确度可达 91%，RNN-RADD 模型的准确度只有 40%。

注意：

- 训练模型的损失函数都是 `sigmoid`
- 在 RNN-RADD 模型中，准确度的计算方式为：序列中的每一个元素与它的标注之间计算准确度。举例：「中国农业银行」，应该是「农行」，标注序列是001001，如果模型计算的结果为（0.6,0.3,0.7,0.3,0.6,0.4），则输出为「中农银」（>0.5），结果显然不对，但是准确度却还是 50%。
- 如果认为输出序列和标注一样才算正确的话，模型在测试集上的准确度为 13%。

## Project Structure

- `data/`, all sorts of datasets
    + `data/men2ent.txt`, the original dataset which contains **2,871,805** lines of mention-entity pairs
- `model/`, trained models
- `out/`, output results
- `temp/`, temporary data
- `filter.py`, remove the bad data from the original dataset
- `sampling.py`, sampling from the output
- `train.log`, log of the training process

## Pre-processing

To generate the `ent2abb` pairs dataset.

## Word Embedding

To train a word embedding model.

1. Change the encoding of the data set to UTF-8:
```
iconv -c -f gbk -t utf8 [in_file] > [out_file]
```

2. Use `pre_process_sougo.py` to convert `xml` into pure `txt` with an article per line,
and remove all non-chinese characters and separate the articles by character
```
python3 pre_process_sougo.py [in_file] [out_file]
```

3. Convert all traditional Chinese characters into simplified Chinese with:
```
opencc -i [in_file] -o [out_file]( -c zht2zhs.ini)
```

4. Train `word2vec` model with `train_word2vec_model.py`
```
python3 train_word2vec_model.py [in_file] [saved_model]
```

## 存在问题

### 加 concept

使用的是 `cn.pbdata.con.txt` 这个文件，里面格式大概是这样的：

```
['信用卡', '签证', '61250', 'credit card', 'visa']
['基本信息', '地址', '22303', 'basic information', 'address']
['基本信息', '电话号码', '22153', 'basic information', 'phone number']
['事件', '学校活动', '22112', 'event', 'school event']
['页', '问题', '11585', 'page', 'questions']
...
```

第一列是 concept，第二列是 word。

随便看了一些例子，觉得这里的 concept 不适合用在模型里，原因有：

1. 对于一个 word，可能会有很多 concepts，比如「智利」就有 501 个 concepts
2. 这些 concepts 可以提供的信息对这个任务可能没有什么帮助，因为有些 concepts 太奇怪了，比如「智利」的几个：
```
['布宜诺斯艾利斯美。', '智利', '1', 'buenos aires-u.s.   ally', 'chile']
['brian@brianmcdaniel.com许多流行的冲浪胜地', '智利', '1', 'brian@brianmcdaniel.com many popular surfing destination', 'chile']
['边疆', '智利', '1', 'borderland', 'chile']
['粗犷气息', '智利', '1', 'bold flavor', 'chile']
['最大的葡萄酒生产国', '智利', '1', 'biggest wine-producing country', 'chile']
['大城市', '智利', '1', 'big city', 'chile']
['最得天独厚的国家', '智利', '1', 'best-endowed country', 'chile']
['豆盘', '智利', '1', 'bean dish', 'chile']
['独裁政府', '智利', '1', 'autocratic government', 'chile']
['与会者', '智利', '1', 'attendees', 'chile']
['联系国', '智利', '1', 'associate country', 'chile']
['有抱负的社会', '智利', '1', 'aspiring society', 'chile']
['亚太经济', '智利', '1', 'asia-pacific economy', 'chile']
['干旱矿区', '智利', '1', 'arid mining area', 'chile']
['安第斯国家', '智利', '1', 'andes country', 'chile']
['美国国家', '智利', '1', 'american nation', 'chile']
['美国食品厂', '智利', '1', 'american food plant', 'chile']
['农业资源丰富的国家', '智利', '1', 'agriculturally-rich country', 'chile']
['农产品出口', '智利', '1', 'agricultural exporter', 'chile']
['非洲和拉丁美洲的国家', '智利', '1', 'african and latin american country', 'chile']
['富裕国家', '智利', '1', 'affluent country', 'chile']
['先进经济体', '智利', '1', 'advanced economy', 'chile']
['先进国家', '智利', '1', 'advanced country', 'chile']
['“新的”生产国', '智利', '1', "``new'' producer country", 'chile']
['“自然”灾害', '智利', '1', "``natural'' disaster", 'chile']
['“中间路线的”国家', '智利', '1', "``middle-of-the-road'' country", 'chile']
['“酷”点', '智利', '1', "``cool'' spot", 'chile']
['5F- 30国家', '智利', '1', '5f-30 country', 'chile']
['第三世界国家', '智利', '1', '3rd world country', 'chile']
['25sub全局和局部的评估', '智利', '1', '25sub-global and local assessment', 'chile']
```

3. 很多 word 没有 concept，我随便试了「中国」，「上海」，「北京」都没有。

### 字数 feature

字数 feature 按照通用的方法加其实没有作用，因为 evaluation 的时候字数这个 feature 的输入永远是 0，所以在 one-hot 编码下之前学到关于字数这个 feature 的参数只有字数为 0 的那些参与了计算，之前学习的根本没用上。

### 评价方法
用搜索引擎判断简称是否是对的听起来倒是挺合理的，可能可行，后续工作可以试一下。

### 1-n, n-1
对于数据中存在的多个 entity 对应一个 abbreviation 的情况，直接扔掉。

对于数据中存在的一个 entity 有多个 abbreviation 的情况（4k 左右个），把它们用来做测试集，其他一对一的数据用来做训练集和验证集，这样模型跑出来的简称只要与测试集的多个简称之一相同就认为正确。

数据规模为：

- Train: 34033
- validation: 3781
- evaluation: 3938

准确率为 40% 上下。
