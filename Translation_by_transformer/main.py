import tensorflow
from tensorflow.keras.models import load_model
from keras_transformer import get_model, decode
import numpy as np
import jieba
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Tips:代码是直接从jupyter notebook粘过来的，没考虑代码结构，直接一条龙处理

file_list = '../../document/en-zh/'


def get_data_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()
        return data


def data_cleaning(data, is_zh=1):
    """
    中文去除句子中的空格
    中文文末标点
    中文分词
    英文转义字符 --> 转义字符也可编辑进词向量，可以不用处理
    英文全部转小写
    英文标点符号前增加空格方便后续处理
    """
    if is_zh:
        data = data.replace(' ', '').split('\n')
        for i in range(len(data)):
            try:
                if data[i][-1] not in ['。', '!', '?', '…']:
                    data[i] += '。'
                data[i] = " ".join(jieba.cut(data[i]))
            except LookupError:
                pass
    else:
        data = data.split('\n')
        try:
            for i in range(len(data)):
                data[i] = data[i][0].lower() + data[i][1:]
                data[i] = data[i].replace('.', ' .').replace(',', ' ,').replace('!', ' !') \
                    .replace('?', ' ?').replace('"', ' "').replace(';', ' ;')
        except LookupError:
            pass
    return data


def handle_super_long(data_zh, data_en):
    """
    处理超长句子
    """
    if len(data_zh) == len(data_en):
        flag = 0
        for i in range(len(data_zh)):
            if len(data_zh[i - flag]) > 80:
                data_zh.pop(i - flag)
                data_en.pop(i - flag)
                flag += 1
    else:
        raise Exception(len(data_zh), len(data_en))
    return data_zh, data_en


# 从文件读取中文数据
train_zh = get_data_from_file(file_list + 'train.zh')
# 清洗中文数据
train_zh = data_cleaning(train_zh, 1)
# 从文件读取英文数据
train_en = get_data_from_file(file_list + 'train.en')
# 清洗英文数据
train_en = data_cleaning(train_en, 0)
# 删除超长句子，删除比例小于0.03%
train_zh, train_en = handle_super_long(train_zh, train_en)
# 将字符串转成列表形式
source_tokens = [sentence.split(' ') for sentence in train_en]
target_tokens = [sentence.split(' ') for sentence in train_zh]


def build_token_dict(token_list):
    """
    生成词典并限制词典大小
    """
    token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<OOV>:': 3
    }
    tokens_all = []
    for tokens in token_list:
        tokens_all += tokens
    token_counter = Counter(tokens_all)
    token_counter = token_counter.most_common(32000)
    for item in token_counter:
        token_dict[item[0]] = len(token_dict)
    return token_dict


def build_input(tokens, dict_, mode):
    """
    构建句子到词典下标的映射
    处理OOV
    """
    input_ = []
    for token in tokens:
        tmp = []
        for item in token:
            if mode == 0:
                try:
                    tmp.append(dict_[item])
                except:
                    tmp.append(3)
            else:
                try:
                    tmp.append([dict_[item]])
                except:
                    tmp.append([3])
        input_.append(tmp)
    return input_


# 构建词典
source_token_dict = build_token_dict(source_tokens)
target_token_dict = build_token_dict(target_tokens)
target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

# 增加开始结束标签
encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]

# Padding到相同长度
source_max_len = max(map(len, encode_tokens))
target_max_len = max(map(len, decode_tokens))
encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

# 转成最终的输入和teacher forcing格式
encode_input = build_input(encode_tokens, source_token_dict, 0)
decode_input = build_input(decode_tokens, target_token_dict, 0)
decode_output = build_input(output_tokens, target_token_dict, 1)

# 生成transformer模型
model = get_model(
    token_num=max(len(source_token_dict), len(target_token_dict)),
    embed_dim=120,
    encoder_num=4,
    decoder_num=4,
    head_num=4,
    hidden_dim=128,
    dropout_rate=0.05,
    attention_activation='relu',
    embed_trainable=True,
    use_same_embed=False,
)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(
    x=[np.array(encode_input), np.array(decode_input)],
    y=np.array(decode_output),
    epochs=15,
    batch_size=8,
)
# 保存训练好的参数
# model.save_weights('./save_weights_30epochs/')
# 使用该语句加载训练好的模型参数
# model.load_weights('./save_weights_30epochs/')

decoded = decode(
    model,
    encode_input,
    start_token=target_token_dict['<START>'],
    end_token=target_token_dict['<END>'],
    pad_token=target_token_dict['<PAD>'],
)

# 计算模型bleu值
smooth = SmoothingFunction()
score = 0
for _ in range(len(decoded)):
    pre = ''.join(map(lambda x: target_token_dict_inv[x], decoded[_][1:-1]))
    target = train_zh[_].replace(' ', '')
    score += sentence_bleu([pre], target, smoothing_function=smooth.method1)
print(score/len(decoded))

# 单条句子翻译应用测试
while True:
    sentence = input('Input english sentence:         ')
    sentence = data_cleaning(sentence, 0)[0].split(' ')
    sentence = ['<START>'] + sentence + ['<END>']
    sentence = sentence + ['<PAD>'] * (source_max_len - len(sentence))
    sentence = build_input([sentence], source_token_dict, 0)[0]
    decoded = decode(
        model,
        sentence,
        start_token=target_token_dict['<START>'],
        end_token=target_token_dict['<END>'],
        pad_token=target_token_dict['<PAD>'],
    )
    print(''.join(map(lambda x: target_token_dict_inv[x], decoded[1:-1])))
