import argparse
import os
import time
import traceback

# import mlflow
from pathlib import Path

import numpy as np
import pandas as pd

from model.detect import detect
from model.train import train
from dataset import Dataset
from utils.dataset_eval import Dataset_eval
from utils.eval import cal_best_PRF


def main(dataset, n_epochs=20, batch_size=64, lr=0.0005, b1=0.5, b2=0.999, seed=None, hidden_dim=64, GAT_heads=4,
         decoder_num_layers=2, TF_styles: str = 'FAP'):
    '''
    :param dataset: instance of Dataset
    :param n_epochs:  number of epochs of training
    :param batch_size:
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
    if TF_styles not in ['AN', 'PAV', 'FAP']:
        raise Exception('"TF_styles" must be a value in ["AN","PAV", "FAP"]')

    gat_ae = train(dataset, n_epochs, batch_size, lr, b1, b2, seed, hidden_dim, GAT_heads, decoder_num_layers,
                   TF_styles)

    trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = detect(gat_ae, dataset,
                                                                                                  batch_size)

    return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--mode', type=str, default='eval', help='specify the mode')
    parser.add_argument('--TF', type=str, default='FAP', help='specify the teacher forcing style')
    parser.add_argument('--beta', type=float, default=0.01, help='specify the hyperparameter beta')

    args = parser.parse_args()

    if args.mode != 'eval':
        attr_keys = ['concept:name', 'org:resource', 'org:role']
        threshold = 0.99

        ROOT_DIR = Path(__file__).parent
        logPath = os.path.join(ROOT_DIR, 'BPIC20_PrepaidTravelCost.xes')
        dataset = Dataset(logPath, attr_keys, args.beta)

        trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = main(dataset,
                                                                                                    n_epochs=20,
                                                                                                    lr=0.0005,
                                                                                                    decoder_num_layers=2,
                                                                                                    batch_size=64,
                                                                                                    hidden_dim=64,
                                                                                                    TF_styles=args.TF)
        attr_level_detection = (attr_level_abnormal_scores > threshold).astype('int64')
        event_level_detection = ((attr_level_abnormal_scores > threshold).sum(axis=2) >= 1).astype('int64')
        trace_level_detection = ((attr_level_abnormal_scores > threshold).sum(axis=(1, 2)) >= 1).astype('int64')

        print(attr_level_detection)
        print(event_level_detection)
        print(trace_level_detection)

    else:
        filePath = 'eventlogs'
        resPath = 'result.csv'
        dataset_names = os.listdir(filePath)
        if 'cache' in dataset_names:
            dataset_names.remove('cache')

        dataset_names_syn = [name for name in dataset_names if (
                'gigantic' in name
                or 'huge' in name
                or 'large' in name
                or 'medium' in name
                or 'p2p' in name
                or 'paper' in name
                or 'small' in name
                or 'wide' in name
        )]
        dataset_names_syn.sort()

        dataset_names_real = list(set(dataset_names) - set(dataset_names_syn))
        dataset_names_real.sort()

        dataset_names = dataset_names_syn + dataset_names_real

        for dataset_name in dataset_names:
            try:
                print(dataset_name)
                start_time = time.time()
                dataset = Dataset_eval(dataset_name, beta=args.beta)
                trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = main(dataset,
                                                                                                            n_epochs=20,
                                                                                                            lr=0.0005,
                                                                                                            decoder_num_layers=2,
                                                                                                            hidden_dim=64,
                                                                                                            batch_size=64,
                                                                                                            TF_styles=args.TF)

                end_time = time.time()
                run_time = end_time - start_time

                print('run_time')
                print(run_time)

                ##trace level
                trace_p, trace_r, trace_f1, trace_aupr = cal_best_PRF(dataset.case_target, trace_level_abnormal_scores)
                print("Trace-level anomaly detection")
                print(f'precision: {trace_p}, recall: {trace_r}, F1-score: {trace_f1}, AP: {trace_aupr}')


                ##event level
                eventTemp = dataset.binary_targets.sum(2).flatten()
                eventTemp[eventTemp > 1] = 1
                event_p, event_r, event_f1, event_aupr = cal_best_PRF(eventTemp, event_level_abnormal_scores.flatten())
                print("Event-level anomaly detection")
                print(f'precision: {event_p}, recall: {event_r}, F1-score: {event_f1}, AP: {event_aupr}')

                ##attr level
                attr_p, attr_r, attr_f1, attr_aupr = cal_best_PRF(dataset.binary_targets.flatten(),
                                                                  attr_level_abnormal_scores.flatten())

                print("Attribute-level anomaly detection")
                print(f'precision: {attr_p}, recall: {attr_r}, F1-score: {attr_f1}, AP: {attr_aupr}')

                datanew = pd.DataFrame(
                    [{'index': dataset_name, 'trace_p': trace_p, "trace_r": trace_r, 'trace_f1': trace_f1,
                      'trace_AP': trace_aupr,
                      'event_p': event_p, "event_r": event_r, 'event_f1': event_f1, 'event_AP': event_aupr,
                      'attr_p': attr_p, "attr_r": attr_r, 'attr_f1': attr_f1, 'attr_AP': attr_aupr,'time':run_time
                      }])
                if os.path.exists(resPath):
                    data = pd.read_csv(resPath)
                    data = data.append(datanew, ignore_index=True)
                else:
                    data = datanew
                data.to_csv(resPath, index=False)
            except:
                traceback.print_exc()
                datanew = pd.DataFrame([{'index': dataset_name}])
                if os.path.exists(resPath):
                    data = pd.read_csv(resPath)
                    data = data.append(datanew, ignore_index=True)
                else:
                    data = datanew
                data.to_csv(resPath, index=False)