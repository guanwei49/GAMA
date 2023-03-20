import numpy as np
import torch
from tqdm import tqdm

from model import device


def detect(gat_ae, dataset):
    gat_ae.eval()
    with torch.no_grad():
        attribute_dims=dataset.attribute_dims
        attr_Shape = (dataset.num_cases, dataset.max_len, dataset.num_attributes)
        attr_level_abnormal_scores=np.zeros(attr_Shape)

        Xs = []
        for i, dim in enumerate(dataset.attribute_dims):
            Xs.append(torch.LongTensor(dataset.features[i]))
        for k, X in enumerate(Xs):
            Xs[k] = X.to(device)

        print("*" * 10 + "detecting" + "*" * 10)
        for index, graphs in enumerate(tqdm(dataset.trace_graphs)):
            for k, graph in enumerate(graphs):
                graphs[k] = graph.to(device)
            one_Xs = []
            for k, X in enumerate(Xs):
                one_Xs.append(X[index])

            case_len = len(graphs[0].x)
            attr_reconstruction_outputs = gat_ae(graphs, one_Xs)

            for attr_index in range(len(attribute_dims)):
                attr_reconstruction_outputs[attr_index] = attr_reconstruction_outputs[attr_index].detach().cpu()
                attr_reconstruction_outputs[attr_index]= torch.softmax(attr_reconstruction_outputs[attr_index],dim=1)


            for time_step in range(1,case_len-1):
                for attr_index in range(len(attribute_dims)):
                    # 取比实际出现的属性值大的其他属性值的概率之和
                    truepos=one_Xs[attr_index][time_step]
                    attr_level_abnormal_scores[index,time_step,attr_index]=attr_reconstruction_outputs[attr_index][time_step,attr_reconstruction_outputs[attr_index][time_step]>attr_reconstruction_outputs[attr_index][time_step,truepos]].sum()
        trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
        return  trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores
