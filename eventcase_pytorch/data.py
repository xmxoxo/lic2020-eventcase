import os
import json

class Corpus(object):
    def __init__(self,sentence,labels):
        self.sentence=sentence
        dicts=dict()
        for one in labels:
            dicts[one[0]]=(one[1],one[2])
        self.labels=dicts

class CorpusProcess(object):
    def __init__(self,train_path, valid_path, test_path):
        self.train_path=train_path
        self.valid_path=valid_path
        self.test_path=test_path

    def _generate_label_id(self,all_corpus):
        """
        :param all_corpus: 通过训练集生成类别词典
        :return: 两个词典
        """
        all_types = []
        for line in all_corpus:
            # print(line.sentence, line.labels)
            for label in line.labels:
                # print(label)
                all_types.append(label)
        labels_type = list(set(all_types))
        # print(labels_type)
        labels_to_id = dict(zip(labels_type, range(len(labels_type))))
        # print(labels_to_id)
        id_to_labels = {value: key for key, value in labels_to_id.items()}
        # print(id_to_labels)
        return labels_to_id, id_to_labels

    def _process_data(self, train_data, labels_to_id):
        """

        :param train_data: 待处理数据集
        :param labels_to_id: 数据集标签化使用的词典
        :return:
        """
        sentences_labels = []
        for line in train_data:
            # print(line.sentence)
            line_label = [one for one in "O" * len(line.sentence)]
            for label in line.labels.items():
                # print(label)
                label_id = labels_to_id[label[0]]
                line_label[label[1][0]] = "B-" + str(label_id)
                for index in range(label[1][0] + 1, label[1][1]):
                    line_label[index] = "I-" + str(label_id)
                # print(label_id)
            # print(line_label)
            sentences_labels.append((line.sentence, line_label))
        return sentences_labels

    def _load_data(self,file_name):
        """
        读入原始数据，抽取text以及事件类型，事件起止下标
        :param file_name: train/valid数据路劲
        :return: Corpus类型的数据list
        """
        D = []
        with open(file_name, "r", encoding="utf-8")as file:
            for line in file:
                line = json.loads(line)
                # print("********************", line["text"])
                arguments = {}

                events = []
                for event in line["event_list"]:
                    # print(event)
                    # print("event_lists---------")

                    for argument in event["arguments"]:
                        ele_start_index = argument["argument_start_index"]
                        ele = argument["argument"]
                        ele_role = argument["role"]
                        type = event["event_type"] + "_" + ele_role
                        # print(type,ele, ele_start_index)
                        events.append((type, ele_start_index, len(ele) + ele_start_index))
                # print(line["text"], events)
                corpus_ele = Corpus(line["text"], events)
                D.append(corpus_ele)
        return D
    def _load_test(self):
        D=[]
        with open(self.test_path, "r", encoding="utf-8")as file:
            for line in file:
                line = json.loads(line)
                D.append(line["text"])
        return D


    def generate_data(self):
        """
        :return:训练集，验证集，测试集 。
        [(sentence,label),(sentence, label ),(sentence, label)]
        sentence|str, label|list
        """

        train_data=self._load_data(self.train_path)
        valid_data = self._load_data(self.valid_path)
        test_data = self._load_test()

        labels_to_id, id_to_labels = self._generate_label_id(train_data)

        train_processed = self._process_data(train_data, labels_to_id)
        valid_processed = self._process_data(valid_data, labels_to_id)
        return train_processed, valid_processed,test_data
