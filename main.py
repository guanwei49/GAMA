import os
import traceback

# import mlflow
from pathlib import Path


from model.detect import detect
from model.train import train
from dataset import Dataset


def main(dataset,n_epochs=2 ,lr=0.0002 ,b1=0.5 ,b2=0.999 ,seed=None ,enc_hidden_dim = 64 , GAT_heads = 4,decoder_num_layers=2, dec_hidden_dim = 64,TF_styles:str='FAP'):
    '''
    :param dataset: instance of Dataset
    :param n_epochs:  number of epochs of training
    :param lr: adam: learning rate
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :param seed: value of Pytorch random seed
    :param enc_hidden_dim: hidden dimensional of encoder_GAT
    :param GAT_heads: heads of first layer of GATs
    :param decoder_num_layers: numberf of layers  of decoder_GRU
    :param dec_hidden_dim: hidden dimensional of decoder_GRU
    :param TF_styles: teacher forcing styles
    :return:
    '''
    if TF_styles not in ['AN','PAV', 'FAP']:
        raise Exception('"TF_styles" must be a value in ["AN","PAV", "FAP"]')

    gat_ae = train(dataset,n_epochs ,lr ,b1 ,b2 ,seed ,enc_hidden_dim, GAT_heads ,decoder_num_layers, dec_hidden_dim ,TF_styles)


    trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores = detect(gat_ae, dataset)

    return  trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores



if __name__ == '__main__':
    attr_keys = ['concept:name', 'org:resource', 'org:role']
    threshold = 0.99

    ROOT_DIR = Path(__file__).parent
    logPath = os.path.join(ROOT_DIR, 'BPIC20_PrepaidTravelCost.xes')
    dataset = Dataset(logPath, attr_keys)

    trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = main(dataset,
                                                                                                n_epochs=2,
                                                                                                lr=0.0002,
                                                                                                decoder_num_layers=2,
                                                                                                enc_hidden_dim=64,
                                                                                                dec_hidden_dim=64,
                                                                                                TF_styles='PAV')
    attr_level_detection = (attr_level_abnormal_scores > threshold).astype('int64')
    event_level_detection = ((attr_level_abnormal_scores > threshold).sum(axis=2) >= 1).astype('int64')
    trace_level_detection = ((attr_level_abnormal_scores > threshold).sum(axis=(1, 2)) >= 1).astype('int64')

    print(attr_level_detection)
    print(event_level_detection)
    print(trace_level_detection)

