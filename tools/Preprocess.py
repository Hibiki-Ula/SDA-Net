import torch
import torch.nn as nn
import os
import json
import numpy as np
from tools import builder
from utils import misc, dist_utils
from utils.misc import process0, savekmeanstxt,savekmeanscentertxt,saveptstxt
from utils.AverageMeter import AverageMeter
import time

def process_data(args, config):
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    n_batches1 = len(train_dataloader)
    n_batches2 = len(test_dataloader)
    batch_time = AverageMeter()
    # print('start train data process............')
    # for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
    #
    #     batch_start_time = time.time()
    #     npoints = config.dataset.train._base_.N_POINTS
    #     dataset_name = config.dataset.train._base_.NAME
    #     if dataset_name == 'PCN':
    #         for i in range(config.total_bs):
    #             pts = data[0][i].cpu().numpy()
    #             ret = process0(pts, config.model.num_group)
    #             savekmeanstxt(ret[0],config.model.num_group,'PCN','train',taxonomy_ids[i] ,model_ids[i])
    #             savekmeanscentertxt(ret[1], ret[2], 'PCN','train',taxonomy_ids[i] ,model_ids[i])
    #
    #     elif dataset_name == 'ShapeNet':
    #         partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)],
    #                                               fixed_points=None)
    #         gt = data.cuda()
    #         partial = partial.cuda()
    #     batch_time.update(time.time() - batch_start_time)
    #     if idx % 10 == 0:
    #         print('[Batch %d/%d] BatchTime = %.3f (s) '% (idx+1, n_batches1, batch_time.val()))
    print('start test data process............')
    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        batch_start_time = time.time()
        npoints = config.dataset.train._base_.N_POINTS
        dataset_name = config.dataset.train._base_.NAME
        if dataset_name == 'PCN':
            pts = data[0].squeeze(0).cpu().numpy()
            ret = process0(pts, config.model.num_group)
            savekmeanstxt(ret[0], config.model.num_group, 'PCN', 'test', taxonomy_ids[0], model_ids[0])
            savekmeanscentertxt(ret[1], ret[2],'PCN', 'test', taxonomy_ids[0], model_ids[0])


        elif dataset_name == 'ShapeNet':
            gt = data.cuda()
            # partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 1 / 4)],
            #                                       fixed_points=None)
            # pts = partial.squeeze().cpu().numpy()
            # pts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'demo',taxonomy_ids[0]+'_'+model_ids[0])
            # np.save(pts_path,pts)
            # pts = gt.squeeze().cpu().numpy()
            pts = gt.cpu().numpy()
            saveptstxt(pts, taxonomy_ids[0], model_ids[0])


        batch_time.update(time.time() - batch_start_time)
        if idx % 10 == 0:
            print('[Batch %d/%d] BatchTime = %.3f (s) ' % (idx + 1, n_batches2, batch_time.val()))

    print('exit............')
