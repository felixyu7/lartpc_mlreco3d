import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml
import time
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ari
from pathlib import Path
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.main_funcs import process_config, train, inference
from mlreco.utils.metrics import *
from mlreco.trainval import trainval
from mlreco.main_funcs import process_config
from mlreco.iotools.factories import loader_factory
from mlreco.main_funcs import cycle

from sklearn.cluster import DBSCAN
from mlreco.utils.ppn import uresnet_ppn_type_point_selector


def make_inference_cfg(train_cfg, gpu=1, snapshot=None, batch_size=1, model_path=None):

    cfg = yaml.load(open(train_cfg, 'r'), Loader=yaml.Loader)
    process_config(cfg)
    inference_cfg = cfg.copy()
    data_keys = inference_cfg['iotool']['dataset']['data_keys']
    
    # Change dataset to validation samples
    data_val = []
    for file_path in data_keys:
        data_val.append(file_path.replace('train', 'test'))
    inference_cfg['iotool']['dataset']['data_keys'] = data_val
    
    # Change batch size to 1 since no need for batching during validation
    inference_cfg['iotool']['batch_size'] = batch_size
    inference_cfg['iotool'].pop('sampler', None)
    inference_cfg['iotool'].pop('minibatch_size', None)
    inference_cfg['trainval']['gpus'] = str(gpu)
    inference_cfg['trainval']["train"] = False
    
    # Analysis keys for clustering
    inference_cfg['model']["analysis_keys"] = {
        "segmentation": 0,
        "clustering": 1,
    }
    
    # Get latest model path if checkpoint not provided.
    if model_path is None:
        model_path = inference_cfg['trainval']['model_path']
    else:
        inference_cfg['trainval']['model_path'] = model_path
    if snapshot is None:
        checkpoints = [int(re.findall('snapshot-([0-9]+).ckpt', f)[0]) for f in os.listdir(
            re.sub(r'snapshot-([0-9]+).ckpt', '', model_path)) if 'snapshot' in f]
        print(checkpoints)
        latest_ckpt = max(checkpoints)
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(str(latest_ckpt)), model_path)
    else:
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(snapshot), model_path)
    inference_cfg['trainval']['model_path'] = model_path
    process_config(inference_cfg)
    return inference_cfg


def main_loop(train_cfg, **kwargs):

    inference_cfg = make_inference_cfg(train_cfg, gpu=kwargs['gpu'], batch_size=1,
                        model_path=kwargs['model_path'])
    start_index = kwargs.get('start_index', 0)
    end_index = kwargs.get('end_index', 20000)
    event_list = list(range(start_index, end_index))
    loader = loader_factory(inference_cfg, event_list=event_list)
    dataset = iter(cycle(loader))
    Trainer = trainval(inference_cfg)
    loaded_iteration = Trainer.initialize()
    output = []

    inference_cfg['trainval']['iterations'] = len(event_list)
    iterations = inference_cfg['trainval']['iterations']

    eps = [1.999, 1.999, 1.999, 1.999]

    for i in event_list:

        print("Iteration: %d" % i)

        data_blob, res = Trainer.forward(dataset)
        semantic_labels = data_blob['cluster_label'][0][:, -1]
        cluster_labels = data_blob['cluster_label'][0][:, 5]
        coords = data_blob['input_data'][0][:, :3]
        index = data_blob['index'][0]

        acc_dict = {}

        for c in (np.unique(semantic_labels)):
            if int(c) >= 4:
                continue
            semantic_mask = semantic_labels == c
            voxels = coords[semantic_mask]
            clabels = cluster_labels[semantic_mask]
            print(index, c)
            if int(c) == 1:
                continue
            else:
                pred = DBSCAN(eps=eps[int(c)], min_samples=1).fit(voxels).labels_
            true_num_clusters = len(np.unique(pred))
            purity, efficiency = purity_efficiency(pred, clabels)
            fscore = 2 * (purity * efficiency) / (purity + efficiency)
            ari = ARI(pred, clabels)
            sbd = SBD(pred, clabels)
            nclusters = len(np.unique(clabels))
            row = (index, c, ari, sbd, purity, efficiency, fscore, \
                nclusters, true_num_clusters)
            output.append(row)
            print("ARI = ", ari)
            # print('SBD = ', sbd)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI', 'SBD',
                'Purity', 'Efficiency', 'FScore', 'num_clusters', 'true_num_clusters'])
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_cfg', '--test_config', help='config_path', type=str)
    #parser.add_argument('-ckpt', '--checkpoint_number', type=int)
    args = parser.parse_args()
    args = vars(args)
    cfg = yaml.load(open(args['test_config'], 'r'), Loader=yaml.Loader)

    train_cfg = cfg['config_path']
    start = time.time()
    output = main_loop(train_cfg, **cfg)
    end = time.time()
    print("Time = {}".format(end - start))
    name = '{}.csv'.format(cfg['name'])
    if not os.path.exists(cfg['target']):
        os.mkdir(cfg['target'])
    target = os.path.join(cfg['target'], name)
    output.to_csv(target, index=False, mode='a', chunksize=50)