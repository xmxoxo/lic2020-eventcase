#! -*- coding: utf-8 -*-
# 百度LIC2020的事件抽取赛道，非官方baseline
# 直接用RoBERTa+CRF
# 在第一期测试集上能达到0.78的F1，优于官方baseline

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm  #进度条封装
import pylcs
import sys
import os

# 基本信息
maxlen = 128
epochs = 20
batch_size = 4
learning_rate = 2e-5
crf_lr_multiplier = 250  # 必要时扩大CRF层的学习率

# bert配置
bert_path = '/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16'

config_path = os.path.join(bert_path, 'bert_config.json')
checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
dict_path = os.path.join(bert_path, 'vocab.txt')


def load_data(filename):
    D = []
    with open(filename,'r',encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            arguments = {}
            for event in l['event_list']:
                for argument in event['arguments']:
                    key = argument['argument']
                    value = (event['event_type'], argument['role'])
                    arguments[key] = value
            D.append((l['text'], arguments))
    return D


# 读取数据,原数据
#train_data[0]
#('雀巢裁员4000人：时代抛弃你时，连招呼都不会打！',
#  {'4000人': ('组织关系-裁员', '裁员人数'), 
#   '雀巢': ('组织关系-裁员', '裁员方')
#  }
#)    
data_path = '../data/'

train_data = load_data(os.path.join(data_path, 'train.json'))
valid_data = load_data(os.path.join(data_path, 'dev.json'))
test_data =  os.path.join(data_path, 'test1.json')
event_data = os.path.join(data_path, 'event_schema.json')

# 读取schema   217个分类   扩展后是 217*2+1个标签,+1是因为有个“O”
# label2id                               id2label
#{                                       {
# ('交往-会见', '会见主体'): 61,           0: ('财经/交易-出售/收购', '时间'),
# ('交往-会见', '会见对象'): 62,           1: ('财经/交易-出售/收购', '出售方'),
# ('交往-会见', '地点'): 60,               2: ('财经/交易-出售/收购', '交易物'),
# ('交往-会见', '时间'): 59,               3: ('财经/交易-出售/收购', '出售价格'),
#}    
with open(event_data,'r',encoding='utf-8') as f:
    id2label, label2id, n = {}, {}, 0
    for l in f:
        l = json.loads(l)
        for role in l['role_list']:
            key = (l['event_type'], role['role'])
            id2label[n] = key
            label2id[key] = n
            n += 1
    num_labels = len(id2label) * 2 + 1

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

#train_data[0]
#('雀巢裁员4000人：时代抛弃你时，连招呼都不会打！',
#  {'雀巢': ('组织关系-裁员', '裁员方'),
#   '4000人': ('组织关系-裁员', '裁员人数')
#  }
#)  
#生成
#batch_token_ids
#【 [101, 7411, 2338, 6161, 1447, 8442, 782, 8038, 3198, 807, 2837,
#  2461, 872, 3198, 8024, 6825, 2875, 1461 6963, 679, 833, 2802, 8013, 102]】
#batch_segment_ids
#【 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0..共115长度]...】
#batch_labels
#【 [0, 363, 364, 0, 0, 365, 366, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...共115长度]...】

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        # is_end是是否是最后一个样本，如果是，则为1
        for is_end, (text, arguments) in self.sample(random):
            '''注意，tokenizer没重写，所以，4000 就存在'''
            #token_ids 
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)   #其实最长的是113，算上前后共115
            labels = [0] * len(token_ids)       #注意，这个labels是有第一位cls和最后一位的
            for argument in arguments.items():
                a_token_ids = tokenizer.encode(argument[0])[0][1:-1]  #雀巢的数字形式
                start_index = search(a_token_ids, token_ids)    #寻找在句子中的位置，+1了
                if start_index != -1:       #如果找到
                    labels[start_index] = label2id[argument[1]] * 2 + 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = label2id[argument[1]] * 2 + 2
            #labels  [0, 363, 364, 0, 0, 365, 366, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

# 搭建模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output = Dense(num_labels)(model.output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)

def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[0].argmax()]


def extract_arguments(text):
    """arguments抽取函数
    """
    #注意这个4000
    '''并没有重写tokenize,所以注意4000人'''
    #text='雀巢裁员4000人：时代抛弃你时，连招呼都不会打！'
    #tokens  ['[CLS]', '雀','巢', '裁', '员', '4000', '人','：','时', '代', '抛', '弃', '你', '时', '，', '连', '招', '呼', '都', '不', '会', '打', '！', '[SEP]']
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)  #把倒数第二个词删掉
    
    #得到映射[[], [0], [1], [2], [3], [4, 5, 6, 7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], []]
    mapping = tokenizer.rematch(text, tokens)
    
    #输入[101, 7411, 2338, 6161, 1447, 8442, 782, 8038, 3198, 807, 2837, 2461, 872, 3198, 8024, 6825, 2875, 1461, 6963, 679, 833, 2802, 8013, 102]
    token_ids = tokenizer.tokens_to_ids(tokens)
    
    #输入[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    segment_ids = [0] * len(token_ids)
    
    #nodes.shape  (24, 435)
    #预测结果 (batch  length  dim)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    
    #(435, 435)
    trans = K.eval(CRF.trans)
    #假设预测labels=[0, 363, 364, 364, 0, 365, 366, 0, 333, 334, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = viterbi_decode(nodes, trans)
    arguments, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            #ch = text[mapping[i][0]:mapping[i][-1] + 1]
            if label % 2 == 1:
                starting = True
                arguments.append([[i], id2label[(label - 1) // 2]])
            elif starting:
                arguments[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False
    #原理，预测label的位置1， mapping中已经把位置编码好了，[0]对应1
    #映射[[], [0], [1], [2], [3], [4, 5, 6, 7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], []]
    #labels=[0, 363, 364, 364, 0, 365, 366, 0, 333, 334, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #arguments为
#    [[[1,2,3], ('组织关系-裁员', '裁员方')],
#     [[5,6], ('组织关系-裁员', '裁员人数')],   
#     [[8,9], ('灾害/意外-坍/垮塌', '时间')],  ]
    #return  
#    {
#    '雀巢裁': ('组织关系-裁员', '裁员方')
#    '4000人': ('组织关系-裁员', '裁员人数'),
#     '时代': ('灾害/意外-坍/垮塌', '时间'),
#     }
    return {
        text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]    : l   
        for w, l in arguments
    }
    
    

#f1, precision, recall = evaluate(valid_data)
# valid_data
#【
#('雀巢裁员4000人：时代抛弃你时，连招呼都不会打！',
#  {
#   '雀巢': ('组织关系-裁员', '裁员方')
#   '4000人': ('组织关系-裁员', '裁员人数'),   
#  }
#),...
#】
def evaluate(data):
    """评测函数（跟官方评测结果不一定相同，但很接近）"""
    X, Y, Z = 1e-10, 1e-10, 1e-10
    #text='雀巢裁员4000人：时代抛弃你时，连招呼都不会打！'
    #arguments={'4000人': ('组织关系-裁员', '裁员人数'), '雀巢': ('组织关系-裁员', '裁员方')}
    for text, arguments in tqdm(data):   #一次验证一个样本
        #{  ('组织关系-裁员', '裁员人数'): '4000人', 
        #   ('组织关系-裁员', '裁员方'): '雀巢'             }
        inv_arguments = {v: k for k, v in arguments.items()}  #真实标签
#pred_arguments:
#  {  '雀巢裁': ('组织关系-裁员', '裁员方'),
#    '4000人': ('组织关系-裁员', '裁员人数'),
#     '时代': ('灾害/意外-坍/垮塌', '时间'),}
        pred_arguments = extract_arguments(text) #从文本预测，得到预测标签
#pred_inv_arguments
#{('灾害/意外-坍/垮塌', '时间'): '时代', 
#('组织关系-裁员', '裁员人数'): '4000人', 
#('组织关系-裁员', '裁员方'): '雀巢裁'}
        pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
        Y += len(pred_inv_arguments)   #预测Y=3
        Z += len(inv_arguments)        #标签Z=2
        for k, v in pred_inv_arguments.items(): #k ('组织关系-裁员', '裁员方') v'雀巢裁'
            if k in inv_arguments:
                # 用最长公共子串作为匹配程度度量
                l = pylcs.lcs(v, inv_arguments[k])   #('雀巢裁','雀巢')   2
                X += 2. * l / (len(v) + len(inv_arguments[k]))  #2*2/(2+3)=0.8如果全对就得1
    #2*0.8/(2+3)     0.8/3     0.8/2
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file,'r',encoding='utf-8') as fr:
        for l in tqdm(fr):
            l = json.loads(l)
#pred_arguments:
#    {
#    '雀': ('组织关系-裁员', '裁员方'),
#    '巢': ('组织关系-裁员', '裁员方'),
#    '裁': ('组织关系-裁员', '裁员方'),     
#    '4000': ('组织关系-裁员', '裁员人数'),
#     '人': ('组织关系-裁员', '裁员人数'), 
#     '时': ('灾害/意外-坍/垮塌', '时间'),
#     '代': ('灾害/意外-坍/垮塌', '时间'),  
#     }
            arguments = extract_arguments(l['text'])
            event_list = []
            for k, v in arguments.items():  #k为'雀',v为('组织关系-裁员', '裁员方')
                event_list.append({
                    'event_type': v[0],
                    'arguments': [{
                        'role': v[1],
                        'argument': k
                    }]
                })
            l['event_list'] = event_list
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()

# 保存模型

class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('model_%d_dev_%.4f.weights' % (epoch,f1))
        print(
            '验证集：f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

#根据时间自动生成文件名
def autoFileName (pre = '',ext = ''):
    import time #+ get_randstr(5)
    filename = ('%s%s%s' % (pre, time.strftime('%Y%m%d%H%M%S',time.localtime())  , ext))
    return filename


if __name__ == '__main__':
    work = 'train'
    if len(sys.argv)>1:
        work = sys.argv[1]

    if work=='train':
        train_generator = data_generator(train_data, batch_size)
        evaluator = Evaluator()

        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
    if work=='predict':
        nfname = ''
        if len(sys.argv)>2:
            nfname = sys.argv[2]
        
        if nfname=='':
            print('usage: ee_1.py predict model.weights' )
            sys.exit()
        
        if os.path.exists(nfname):
            model.load_weights(nfname)
            outfile = autoFileName('./pred_', '.json')
            predict_to_file(test_data, outfile)
            print('save to %s' % outfile)

# CUDA_VISIBLE_DEVICES=1
# 训练： python ee_1.py train
# 预测:  python ee_1.py predict model.weights
