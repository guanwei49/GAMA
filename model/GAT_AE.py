import math
import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Variable
from model import device
from torch_geometric.nn import GATConv


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i

        pe = torch.zeros(max_seq_len, input_dim)
        for pos in range(max_seq_len):
            for i in range(0, input_dim, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / input_dim)))
                if i+1 < input_dim:
                    pe[pos, i + 1] = \
                        math.cos(pos / (10000 ** ((2 * (i + 1)) / input_dim)))
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.input_dim)
        # add constant to embedding
        seq_len = x.size(0)
        pe = Variable(self.pe[ :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class GAT_Encoder(nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, in_head , max_seq_len):
        super().__init__()
        # self.embed = nn.Linear(input_dim, input_dim)

        self.pe = PositionalEncoder(input_dim,max_seq_len)

        self.conv1 = GATConv(in_channels=input_dim,
                             out_channels=enc_hidden_dim,
                             heads=in_head,
                             dropout=0.4)
        self.conv2 = GATConv(in_channels=enc_hidden_dim * in_head,
                             out_channels=enc_hidden_dim,
                             concat=False,
                             heads=1,
                             dropout=0.4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = self.embed(x)
        x = self.pe(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.unsqueeze(1)

        return x   # x:[seq_len,1,enc_hidden_dim]


# class Attention(nn.Module):
#     '''
#     加性注意力
#     '''
#     def __init__(self, enc_hid_dim, dec_hid_dim):
#         super().__init__()
#
#         self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim, bias=False)  # 输出的维度是任意的
#         self.v = nn.Linear(dec_hid_dim, 1, bias=False)  # 将输出维度置为1
#
#     def forward(self, s, enc_output):
#         # s = [batch_size, dec_hidden_dim]
#         # enc_output = [seq_len, batch_size, enc_hid_dim * 2]
#
#         seq_len = enc_output.shape[0]
#
#         # repeat decoder hidden state seq_len times
#         # s = [seq_len, batch_size, dec_hid_dim]
#         s = s.repeat(seq_len, 1,1)  # [batch_size, dec_hid_dim]=>[seq_len, batch_size, dec_hid_dim]
#
#         energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
#
#         attention = self.v(energy).squeeze(
#             2)  # [seq_len, batch_size, dec_hid_dim]=>[seq_len，batch_size, 1] => [seq_len, batch_size]
#
#         attention_probs=F.softmax(attention, dim=0).transpose(0, 1).unsqueeze(1)  # [batch_size, 1 , seq_len]
#
#         enc_output = enc_output.transpose(0, 1)
#
#         # # c = [1, batch_size, enc_hid_dim * 2]
#         c = torch.bmm(attention_probs, enc_output).transpose(0, 1)
#
#         return c,attention_probs


class Attention(nn.Module):
    '''
    点积注意力
    '''
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.hidden=64
        self.query = nn.Linear(dec_hid_dim, self.hidden)
        self.key = nn.Linear(enc_hid_dim , self.hidden)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hidden_dim]
        # enc_output = [seq_len, batch_size, enc_hid_dim * 2]
        s = s.unsqueeze(0) # [batch_size, dec_hid_dim]=>[1, batch_size, dec_hid_dim]
        s=s.transpose(0, 1)
        q=self.query(s)
        enc_output=enc_output.transpose(0, 1)
        k=self.key(enc_output)
        k=k.transpose(1, 2)

        attention_scores= torch.bmm(q, k)
        attention_scores=attention_scores/ math.sqrt(self.hidden)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)##[ batch_size, 1, seq_len]

        result = torch.bmm(attention_probs, enc_output).transpose(0, 1)

        return result,attention_probs

class Decoder_act(nn.Module):
    def __init__(self, vocab_size, enc_hid_dim, dec_hid_dim,num_layers,output_dim):
        super().__init__()
        self.num_layers=num_layers
        self.vocab_size = vocab_size
        self.attention =  Attention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(vocab_size, enc_hid_dim)
        self.rnn = nn.GRU(enc_hid_dim + enc_hid_dim, dec_hid_dim,num_layers = num_layers,dropout=0.3)
        self.fc_out = nn.Linear(enc_hid_dim  + dec_hid_dim + enc_hid_dim , output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, dec_input, s, enc_output):
        # dec_input = [1]
        # s = [1, dec_hid_dim]
        # enc_output = [case_len*num_attr, 1, enc_hid_dim ]

        dec_input = dec_input.unsqueeze(0) # dec_input = [1]=> [1,1]
        dec_input =self.embedding(dec_input) # dec_input = [1,1] => [1,1,enc_hid_dim]

        dropout_dec_input = self.dropout(dec_input) #  [1, 1,enc_hid_dim]=>[1,1,enc_hid_dim]

        # c = [1, 1, enc_hid_dim], attention_probs=[1,1,case_len*num_attr]
        c,attention_probs = self.attention(s, enc_output)

        rnn_input = torch.cat((dropout_dec_input, c), dim = 2) # rnn_input = [1, 1, enc_hid_dim+ enc_hid_dim]

        # dec_output=[1,1,dec_hid_dim]  ; dec_hidden=[num_layers,1,dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.repeat( self.num_layers,1,1))

        dec_output = dec_output.squeeze(0) # dec_output:[ 1, dec_hid_dim]

        c = c.squeeze(0)  # c:[1, enc_hid_dim]

        dropout_dec_input=dropout_dec_input.squeeze(0)  # dec_input:[1, enc_hid_dim]

        pred = self.fc_out(torch.cat((dec_output, c, dropout_dec_input), dim = 1))# pred = [1, output_dim]

        return pred, dec_hidden[-1],attention_probs

class Decoder_attr(nn.Module):
    def __init__(self, vocab_size, enc_hid_dim, dec_hid_dim,num_layers,output_dim,TF_styles):
        super().__init__()
        self.num_layers=num_layers
        self.vocab_size = vocab_size
        self.attention =  Attention(enc_hid_dim, dec_hid_dim)
        self.embedding_act = nn.Embedding(vocab_size, enc_hid_dim)
        self.embedding_attr = nn.Embedding(output_dim, enc_hid_dim)
        self.TF_styles=TF_styles
        emb_num = 1
        if TF_styles == 'FAP' :
            emb_num=2
        self.rnn = nn.GRU(enc_hid_dim * emb_num + enc_hid_dim, dec_hid_dim,num_layers = num_layers,dropout=0.3)
        self.fc_out = nn.Linear(enc_hid_dim  + dec_hid_dim + enc_hid_dim * emb_num , output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, dec_input_act,dec_input_attr, s, enc_output):
        # dec_input_act = [1]
        # dec_input_attr = [1]
        # s = [1, dec_hid_dim]
        # enc_output = [case_len*num_attr, 1, enc_hid_dim * 2]

        dec_input_act = dec_input_act.unsqueeze(0) # dec_input = [1]=> [1,1]
        dec_input_act =self.embedding_act(dec_input_act) # dec_input = [1, 1] => [1, 1,enc_hid_dim]

        dropout_dec_input_act = self.dropout(dec_input_act) #  [1, 1,enc_hid_dim]=>[1,1,enc_hid_dim]

        dec_input_attr = dec_input_attr.unsqueeze(0)  # dec_input = [1, 1]
        dec_input_attr = self.embedding_attr(dec_input_attr)  # dec_input = [1, 1] => [1, 1,enc_hid_dim]

        dropout_dec_input_attr = self.dropout(dec_input_attr)  # [1, 1,enc_hid_dim]=>[1,1,enc_hid_dim]

        # c = [1, 1, enc_hid_dim], attention_probs=[1,1,case_len*num_attr]
        c,attention_probs = self.attention(s, enc_output)

        if  self.TF_styles=='AN':
            rnn_input = torch.cat((dropout_dec_input_act,  c),
                                  dim=2)  # rnn_input = [1, batch_size, enc_hid_dim + enc_hid_dim]
        elif  self.TF_styles=='PAV':
            rnn_input = torch.cat((dropout_dec_input_attr, c),
                                  dim=2)  # rnn_input = [1, batch_size, enc_hid_dim+ enc_hid_dim]
        else:  #FAP
            rnn_input = torch.cat((dropout_dec_input_act, dropout_dec_input_attr, c),
                                  dim=2)  # rnn_input = [1, batch_size, (enc_hid_dim * 2)+ enc_hid_dim]


        dec_output, dec_hidden = self.rnn(rnn_input, s.repeat( self.num_layers,1,1))
        # dec_output=[1,1,dec_hid_dim]  ; dec_hidden=[num_layers,1,dec_hid_dim]
        dec_output = dec_output.squeeze(0) # dec_output:[ batch_size, dec_hid_dim]

        c = c.squeeze(0)  # c:[1, enc_hid_dim]

        dropout_dec_input_act=dropout_dec_input_act.squeeze(0)  # dropout_dec_input_act:[1, enc_hid_dim]
        dropout_dec_input_attr=dropout_dec_input_attr.squeeze(0) # dropout_dec_input_attr:[1, enc_hid_dim]

        if self.TF_styles == 'AN':
            pred = self.fc_out(torch.cat((dec_output, c, dropout_dec_input_act), dim = 1)) # pred = [1, output_dim]
        elif self.TF_styles == 'PAV':
            pred = self.fc_out(torch.cat((dec_output, c,  dropout_dec_input_attr), dim=1)) # pred = [1, output_dim]
        else:  # FAP
            pred = self.fc_out(torch.cat((dec_output, c, dropout_dec_input_act,dropout_dec_input_attr), dim = 1)) # pred = [1, output_dim]

        return pred, dec_hidden[-1],attention_probs

class GAT_AE(nn.Module):
    def __init__(self,  attribute_dims,max_seq_len ,enc_hidden_dim, GAT_heads, decoder_num_layers, dec_hidden_dim,TF_styles):
        super().__init__()
        encoders=[]
        decoders=[]
        self.attribute_dims=attribute_dims
        for i, dim in enumerate(attribute_dims):
            encoders.append( GAT_Encoder(int(dim), enc_hidden_dim, GAT_heads, max_seq_len ))
            if i == 0:
                decoders.append(Decoder_act(int(attribute_dims[0] + 1), enc_hidden_dim, dec_hidden_dim, decoder_num_layers,
                                            int(dim + 1)))
            else:
                decoders.append(
                    Decoder_attr(int(attribute_dims[0] + 1), enc_hidden_dim, dec_hidden_dim, decoder_num_layers,
                                 int(dim + 1),TF_styles))
        self.encoders=nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)


    def forward(self, graphs , Xs):
        '''
        :param graphs:是多个属性对应的图，每一个属性作为一个graph
        :param Xs:是多个属性，每一个属性作为一个X ： [attribute_dims, time_step]
        :return:
        '''
        case_len = len(graphs[0].x)
        attr_reconstruction_outputs = [] #概率分布 probability map
        s = []  #解码层GRU初始隐藏表示
        enc_output = None
        # Z=None
        for i, dim in enumerate(self.attribute_dims):

            output_dim = int(dim) + 1
            graph = graphs[i]
            X=Xs[i]

            attr_reconstruction_outputs.append(torch.zeros(case_len, output_dim).to(device))  # 存储decoder的所有输出
            enc_output_ = self.encoders[i](graph)
            s_= enc_output_.mean(0)  #取所有节点的平均作为decoder的第一个隐藏状态的输入
            # enc_output_ = [case_len , 1, enc_hid_dim]
            if enc_output is None:
                # Z = enc_output_.squeeze(1)
                enc_output = enc_output_
            else:
                # Z = torch.cat((Z, enc_output_.squeeze(1)), dim=1)
                enc_output = torch.cat((enc_output, enc_output_), dim=0)
            # enc_output = [case_len*len(self.attribute_dims), 1, enc_hid_dim ]
            s.append(s_)

            attr_reconstruction_outputs[-1][0, X[0]] = 1

        for i, dim in enumerate(self.attribute_dims):
            if i == 0:
                X = Xs[i]
                s0 = s[i]
                dec_input = X[0].unsqueeze(0)  # target的第一列，即是起始字符 teacher_forcing

                for t in range(1, case_len):
                    dec_output, s0, attention_probs = self.decoders[i](dec_input, s0, enc_output)

                    # 存储每个时刻的输出
                    attr_reconstruction_outputs[i][t] = dec_output

                    dec_input = X[t].unsqueeze(0)  # teacher_forcing
            else:
                s0 = s[i]
                X_act = Xs[0]  # activity
                X_attr = Xs[i]
                dec_input_attr = X_attr[0].unsqueeze(0)  # target的第一列，即是起始字符 teacher_forcing

                for t in range(1, case_len):
                    dec_input_act = X_act[t].unsqueeze(0)  # teacher_forcing activity

                    dec_output, s0, attention_probs = self.decoders[i](dec_input_act, dec_input_attr, s0, enc_output)  # s0隐藏状态

                    # 存储每个时刻的输出
                    attr_reconstruction_outputs[i][t] = dec_output

                    dec_input_attr = X_attr[t].unsqueeze(0)  # teacher_forcing



        return attr_reconstruction_outputs
