from data import CorpusProcess
import os

data_path = 'D:/DataBase/nlp_data/baidu_/event_role/all_data/'
train_path=os.path.join(data_path+"train.json")
valid_path=os.path.join(data_path+"dev.json")
test_path=os.path.join(data_path+"test1.json")

corpus=CorpusProcess(train_path, valid_path, test_path)

train_data,valid_data,test_data=corpus.generate_data()

print("共读入 训练集: {},验证机: {}, 测试集: {}".format(len(train_data),len(valid_data),len(test_data)))
print("实例"+"*  "*20)
print("train sample: ", train_data[0])
print("句子长度为 {},标签长度为 {}".format(len(train_data[0][0]),len(train_data[0][1])))
print("test sample: ", test_data[0])
