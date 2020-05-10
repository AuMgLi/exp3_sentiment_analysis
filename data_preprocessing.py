import torch
import random
from torchtext import data
from torchtext import datasets

SEED = 2020
torch.manual_seed(SEED)

# 创建两个Field对象：这两个对象包含了我们打算如何预处理文本数据的信息
# spaCy:英语分词器,类似于NLTK库，如果没有传递tokenize参数，则默认只是在空格上拆分字符串
TEXT = data.Field(tokenize='spacy')
# LabelField是Field类的一个特殊子集，专门用于处理标签
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)  # 下载IMBb数据集
# print(f'Number of training examples: {len(train_data)}')  # 25000
# print(f'Number of testing examples: {len(test_data)}')  # 25000
# print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(split_ratio=0.7, random_state=random.seed(SEED))  # 分割出验证集
# print(f'Number of training examples: {len(train_data)}')  # 17500
# print(f'Number of validation examples: {len(valid_data)}')  # 7500
# print(f'Number of testing examples: {len(test_data)}')  # 25000

# 创建vocabulary，将每个单词映射到一个数字
TEXT.build_vocab(train_data, max_size=25000)  # 只保留顶部（最常见的）25000个单词，其他单词以<unk>表示
LABEL.build_vocab(train_data)
# print(f'Unique tokens in TEXT vocab.: {len(TEXT.vocab)}')  # 25002 (additional tokens <unk> and <pad>)
# print(f'Unique tokens in LABEL vocab.: {len(LABEL.vocab)}')  # 2
# print(TEXT.vocab.freqs.most_common(10))  # 最常见10个单词及其频次
# print(LABEL.vocab.stoi)  # {'neg': 0, 'pos': 1}


def get_imdb_data_iterators(device, batch_size=64):
    # 创建iterators，每个iteration都会返回一个batch的examples
    # BucketIterator会把长度差不多的句子放到同一个batch中，确保每个batch中不出现太多的padding
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device,
    )

    return train_iterator, valid_iterator, test_iterator
