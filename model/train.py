import torch
from torch import nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from model import device
from model.GAT_AE import  GAT_AE
import random
def train(dataset,n_epochs,batch_size,lr ,b1 ,b2 ,seed,hidden_dim , GAT_heads , decoder_num_layers ,TF_styles):
    if type(seed) is int:
        torch.manual_seed(seed)

    gat_ae = GAT_AE(dataset.attribute_dims, dataset.max_len, hidden_dim, GAT_heads, decoder_num_layers, TF_styles)
    loss_func = nn.CrossEntropyLoss()

    gat_ae.to(device)

    optimizer = torch.optim.Adam(gat_ae.parameters(),lr=lr, betas=(b1, b2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    Xs = []
    for i, dim in enumerate(dataset.attribute_dims):
        Xs.append(torch.LongTensor(dataset.features[i]))

    print("*"*10+"training"+"*"*10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        #自定义的dataloader
        indexs = [i for i in range(len(dataset))]  #打乱顺序
        random.shuffle(indexs)

        for bathc_i in tqdm(range(batch_size, len(indexs)+1,batch_size)):
            this_batch_indexes=indexs[bathc_i-batch_size:bathc_i]
            nodes_list = [dataset.node_xs[i] for i in this_batch_indexes]
            edge_indexs_list = [dataset.edge_indexs[i] for i in this_batch_indexes]
            Xs_list=[]
            graph_batch_list = []
            for i in range(len(dataset.attribute_dims)):
                Xs_list.append(Xs[i][this_batch_indexes].to(device))
                graph_batch = Batch.from_data_list([Data(x=nodes_list[b][i], edge_index=edge_indexs_list[b])
                                                    for b in range(len(nodes_list))])
                graph_batch_list.append(graph_batch.to(device))
            mask= torch.tensor(dataset.mask[this_batch_indexes]).to(device)


            attr_reconstruction_outputs = gat_ae(graph_batch_list,Xs_list,mask,len(this_batch_indexes))

            optimizer.zero_grad()

            loss=0.0
            mask[:, 0] = False # 除了每一个属性的起始字符之外,其他重建误差
            for i in range(len(dataset.attribute_dims)):
                #--------------
                # 除了每一个属性的起始字符之外,其他重建误差
                #---------------
                pred=attr_reconstruction_outputs[i][mask]
                true=Xs_list[i][mask]
                loss+=loss_func(pred,true)

            train_loss += loss.item()
            train_num += 1
            loss.backward()
            optimizer.step()
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch=train_loss / train_num
        print(f"[Epoch {epoch+1:{len(str(n_epochs))}}/{n_epochs}] "
              f"[loss: {train_loss_epoch:3f}]")
        scheduler.step()

    return gat_ae

