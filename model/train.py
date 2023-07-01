import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from model import device
from model.GAT_AE import  GAT_AE
import random
def train(dataset,n_epochs ,lr ,b1 ,b2 ,seed,hidden_dim , GAT_heads , decoder_num_layers,TF_styles):

    if type(seed) is int:
        torch.manual_seed(seed)

    gat_ae = GAT_AE(dataset.attribute_dims, dataset.max_len, hidden_dim, GAT_heads, decoder_num_layers, TF_styles)
    loss_func = nn.CrossEntropyLoss()

    gat_ae.to(device)

    optimizer = torch.optim.Adam(gat_ae.parameters(),lr=lr, betas=(b1, b2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    Xs=[]
    for i, dim in enumerate(dataset.attribute_dims):
        Xs.append( torch.LongTensor(dataset.features[i]))
    for k, X in enumerate(Xs):
        Xs[k] = X.to(device)

    print("*"*10+"training"+"*"*10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        indexs = [i for i in range(len(dataset))]  #打乱顺序
        random.shuffle(indexs)

        for index in tqdm(indexs):
            graphs=dataset.trace_graphs[index]
            for k, graph in enumerate(graphs):
                graphs[k] = graph.to(device)
            one_Xs=[]

            for k, X in enumerate(Xs):
                one_Xs.append(X[index])

            case_len = len(graphs[0].x)
            attr_reconstruction_outputs = gat_ae(graphs,one_Xs)

            optimizer.zero_grad()

            loss=0.0
            for ij in range(len(dataset.attribute_dims)):
                #--------------
                # 除了每一个属性的起始字符之外,其他重建误差
                #---------------
                pred=attr_reconstruction_outputs[ij][1:,:]
                true=one_Xs[ij][1:case_len]
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

