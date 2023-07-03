import numpy as np
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from model import device


def detect(gat_ae, dataset, batch_size):
    gat_ae.eval()

    with torch.no_grad():
        final_res = []
        attribute_dims=dataset.attribute_dims

        Xs = []
        for i, dim in enumerate(dataset.attribute_dims):
            Xs.append(torch.LongTensor(dataset.features[i]))

        print("*" * 10 + "detecting" + "*" * 10)

        pre = 0

        for bathc_i in tqdm(range(batch_size, len(dataset)+batch_size, batch_size)):
            if bathc_i <= len(dataset):
                this_batch_indexes = list(range(pre, bathc_i))
            else:
                this_batch_indexes = list(range(pre,  len(dataset)))

            nodes_list = [dataset.node_xs[i] for i in this_batch_indexes]
            edge_indexs_list = [dataset.edge_indexs[i] for i in this_batch_indexes]
            Xs_list = []
            graph_batch_list = []
            for i in range(len(dataset.attribute_dims)):
                Xs_list.append(Xs[i][this_batch_indexes].to(device))
                graph_batch = Batch.from_data_list([Data(x=nodes_list[b][i], edge_index=edge_indexs_list[b])
                                                    for b in range(len(nodes_list))])
                graph_batch_list.append(graph_batch.to(device))
            mask = torch.tensor(dataset.mask[this_batch_indexes]).to(device)

            attr_reconstruction_outputs = gat_ae(graph_batch_list, Xs_list, mask, len(this_batch_indexes))

            for attr_index in range(len(attribute_dims)):
                attr_reconstruction_outputs[attr_index] = torch.softmax(attr_reconstruction_outputs[attr_index], dim=2)

            this_res = []
            for attr_index in range(len(attribute_dims)):
                # 取比实际出现的属性值大的其他属性值的概率之和
                temp = attr_reconstruction_outputs[attr_index]
                index = Xs_list[attr_index].unsqueeze(2)
                probs = temp.gather(2, index)
                temp[(temp <= probs)] = 0
                res = temp.sum(2)
                res = res * (mask)
                this_res.append(res)

            final_res.append(torch.stack(this_res, 2))

            pre = bathc_i

        attr_level_abnormal_scores = np.array(torch.cat(final_res, 0).detach().cpu())
        trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores

