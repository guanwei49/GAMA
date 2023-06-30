import os
from collections import defaultdict

import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class Dataset(object):
    def __init__(self, dataset_Path,attr_keys, beta=0):
        # Public properties
        self.dataset_name = dataset_Path
        self.trace_graphs = []
        self.node_dims=[]
        self.beta=beta

        start_symbol = '▶'
        end_symbol = '■'

        ######hospital test#######
        logPath = os.path.join( dataset_Path)
        log = xes_importer.apply(logPath)

        self.case_lens = []
        feature_columns = defaultdict(list)
        for trace in log:
            self.case_lens.append(len(trace) + 2)
            for attr_key in attr_keys:
                feature_columns[attr_key].append(start_symbol)
            for event in trace:
                for attr_key in attr_keys:
                    feature_columns[attr_key].append(event[attr_key])
            for attr_key in attr_keys:
                feature_columns[attr_key].append(end_symbol)

        # print(feature_columns)

        for key in feature_columns.keys():
            encoder = LabelEncoder()
            feature_columns[key] = encoder.fit_transform(feature_columns[key]) + 1

        # Transform back into sequences
        case_lens = np.array(self.case_lens)
        offsets = np.concatenate(([0], np.cumsum(case_lens)[:-1]))
        self.features = [np.zeros((case_lens.shape[0], case_lens.max()),dtype=int) for _ in range(len(feature_columns))]
        for i, (offset, case_len) in enumerate(zip(offsets, case_lens)):
            for k, key in enumerate(feature_columns):
                x = feature_columns[key]
                self.features[k][i, :case_len] = x[offset: offset + case_len]

        self._gen_trace_graphs()

    def __len__(self):
        return self.num_cases

    def _gen_trace_graphs(self):

        graph_relation = np.zeros((self.attribute_dims[0]+1,self.attribute_dims[0]+1),dtype='int32')
        for case_index in range(self.num_cases):
            if self.case_lens[case_index]>1:
                for activity_index in range(1, self.case_lens[case_index]):
                    graph_relation[ self.features[0][case_index][activity_index - 1] , self.features[0][case_index][activity_index] ] += 1
        dims_temp = []
        dims_temp.append(self.attribute_dims[0])
        for j in range(1, len(self.attribute_dims)):
            dims_temp.append(dims_temp[j - 1] + self.attribute_dims[j])
        dims_temp.insert(0, 0)
        dims_range = [(dims_temp[i - 1], dims_temp[i]) for i in range(1, len(dims_temp))]

        graph_relation = np.array(graph_relation >= self.beta*self.num_cases, dtype='int32')

        onehot_features = self.flat_onehot_features
        eye = np.eye(self.max_len, dtype=int)
        for case_index in range(self.num_cases):  #生成图
            attr_graphs = []
            edge = []
            xs = []
            ##构造顶点信息
            for attr_index in range(self.num_attributes):
                xs.append(onehot_features[case_index, :self.case_lens[case_index],
                          dims_range[attr_index][0]:dims_range[attr_index][1]])

            if self.case_lens[case_index]>1:
                ##构造边信息
                node = self.features[0][case_index,:self.case_lens[case_index]]
                for activity_index in range(0, self.case_lens[case_index]):
                    out = np.argwhere( graph_relation[self.features[0][case_index,activity_index]] == 1).flatten()
                    a = set(node)
                    b = set(out)
                    if activity_index+1< self.case_lens[case_index]:
                        edge.append([activity_index, activity_index+1])  #保证trace中相连的activity一定有边。
                    for node_name in a.intersection(b):
                        for node_index in np.argwhere(node == node_name).flatten():
                            if  activity_index+1 != node_index:
                                edge.append([activity_index, node_index])  # 添加有向边
            edge_index = torch.tensor(edge, dtype=torch.long)
            for attr_index in range(self.num_attributes):
                attr_graphs.append(Data(torch.tensor(xs[attr_index], dtype=torch.float), edge_index=edge_index.T))
            self.trace_graphs.append(attr_graphs)

        self.node_dims = self.attribute_dims.copy()

    @property
    def num_cases(self):
        """Return number of cases in the event log, i.e., the number of examples in the dataset."""
        return len(self.features[0])

    @property
    def num_events(self):
        """Return the total number of events in the event log."""
        return sum(self.case_lens)

    @property
    def max_len(self):
        """Return the length of the case with the most events."""
        return self.features[0].shape[1]


    @property
    def attribute_dims(self):
        return  np.asarray([int(f.max())  for f in self.features])

    @property
    def num_attributes(self):
        """Return the number of attributes in the event log."""
        return len(self.features)

    @property
    def onehot_features(self):
        """
        Return one-hot encoding of integer encoded features

        As `features` this will return one tensor for each attribute. Shape of tensor for each attribute will be
        (number_of_cases, max_case_length, attribute_dimension). The attribute dimension refers to the number of unique
        values of the respective attribute encountered in the event log.

        :return:
        """
        return [to_categorical(f)[:, :, 1:] for f in self.features]



    @property
    def flat_onehot_features(self):
        """
        Return combined one-hot features in one single tensor.

        One-hot vectors for each attribute in each event will be concatenated. Resulting shape of tensor will be
        (number_of_cases, max_case_length, attribute_dimension[0] + attribute_dimension[1] + ... + attribute_dimension[n]).

        :return:
        """
        return np.concatenate(self.onehot_features, axis=2)
