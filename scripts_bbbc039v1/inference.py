import os
from PIL.Image import fromqimage
import cv2
import h5py
import yaml
import time
import argparse
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from attrdict import AttrDict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data_provider import Provider, Validation
from utils.show import show_affs, val_show, val_show_emd
from utils.utils import setup_seed
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss, BCE_loss_func
from loss.loss_embedding_mse import embedding_loss, embedding2affs
from model.unet2d_residual import ResidualUNet2D_deep as ResidualUNet2D_affs
from utils.seg_mutex import seg_mutex
from utils.lmc import multicut_multi
from utils.seg_waterz import seg_waterz
from utils.affinity_ours import multi_offset
from utils.postprocessing import merge_small_object, remove_samll_object
from utils.merge_small import merge_small_segments
from data.data_segmentation import relabel

from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from utils.metrics_bbbc import agg_jc_index, pixel_f1, remap_label, get_fast_pq, FGBGDice, BestDice

import warnings
warnings.filterwarnings("ignore")

def merge_func(seg, step=4):
    seg = merge_small_object(seg)
    seg = merge_small_object(seg, threshold=25, window=11)
    seg = merge_small_object(seg, threshold=50, window=11)
    seg = merge_small_object(seg, threshold=100, window=21)
    return seg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='bbbc039v1')
    parser.add_argument('-mn', '--model_name', type=str, default=None)
    parser.add_argument('-id', '--model_id', type=str, default=None)
    parser.add_argument('-m', '--mode', type=str, default='validation') # test
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-sw', '--show', action='store_false', default=True)
    parser.add_argument('-se', '--show_embedding', action='store_true', default=False)
    parser.add_argument('-ne', '--norm_embedding', action='store_true', default=False)
    parser.add_argument('-sn', '--show_num', type=int, default=4)
    parser.add_argument('-sd', '--stride', type=int, default=None)
    parser.add_argument('-nb', '--neighbor', type=int, default=None)
    parser.add_argument('-sf', '--shifts', type=str, default=None)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('../inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+args.model_id
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    affs_img_path = os.path.join(out_affs, 'affs_img')
    seg_img_path = os.path.join(out_affs, 'seg_img')
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    device = torch.device('cuda:0')
    model = ResidualUNet2D_affs(in_channels=cfg.MODEL.input_nc,
                                out_channels=cfg.MODEL.output_nc,
                                nfeatures=cfg.MODEL.filters,
                                if_sigmoid=cfg.MODEL.if_sigmoid).to(device)

    ckpt_path = os.path.join('../trained_models', trained_model, args.model_id+'.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    valid_provider = Validation(cfg, mode=args.mode)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False, pin_memory=True)

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")
    criterion_mask = BCE_loss_func

    if args.shifts is None:
        shifts = list(cfg.DATA.shifts)
    else:
        shifts_str = args.shifts
        split = shifts_str.split(',')
        shifts = []
        for k in split:
            shifts.append(int(k))
    if args.neighbor is None:
        neighbor = cfg.DATA.neighbor
    else:
        neighbor = args.neighbor
    print('shifts is', shifts, end=' ')
    print('neighbor is', neighbor)
    offsets = multi_offset(shifts, neighbor=neighbor)
    nb_half = neighbor // 2
    losses_valid = []
    aji_score = []
    dice_score = []
    f1_score = []
    pq_score = []
    diff = []
    all_voi = []
    all_arand = []
    affs = []
    masks = []
    seg = []
    if args.stride is None:
        stride = list(cfg.DATA.strides)
    else:
        stride = [args.stride, args.stride]

    start_time = time.time()
    f_txt = open(os.path.join(out_affs, 'score.txt'), 'w')
    for k, batch in enumerate(val_loader, 0):
        batch_data = batch
        inputs = batch_data['image'].cuda()
        target = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()
        target_ins = batch_data['seg'].cuda()
        affs_mask = batch_data['mask'].cuda()
        with torch.no_grad():
            emd4, emd3, emd2, emd1, embedding, pred_mask = model(inputs)
        loss_embedding, pred, _ = embedding_loss(embedding, target, weightmap, affs_mask, criterion, offsets, affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
        loss_mask = cfg.TRAIN.mask_weight * criterion_mask(pred_mask, torch.gt(target_ins[:, 0], 0), weight_rate=[10, 1]).to(device)
        tmp_loss = loss_embedding + loss_mask
        losses_valid.append(tmp_loss.item())
        pred = F.relu(pred)
        output_affs = np.squeeze(pred.data.cpu().numpy())

        # post-processing
        gt_ins = np.squeeze(batch_data['seg'].numpy()).astype(np.uint16)
        # crop 
        output_affs = output_affs[:, 92:-92, 4:-4]
        affs.append(output_affs.copy())
        gt_ins = gt_ins[92:-92, 4:-4]
        # gt_mask = gt_ins.copy()
        # gt_mask[gt_mask != 0] = 1
        # pred_seg = seg_mutex(output_affs, offsets=offsets, strides=stride, mask=gt_mask).astype(np.uint16)
        if cfg.TRAIN.mask_weight == 0:
            pred_seg = seg_mutex(output_affs, offsets=offsets, strides=list(cfg.DATA.strides)).astype(np.uint16)
        else:
            pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1).squeeze(0)
            pred_mask_b = pred_mask.data.cpu().numpy()
            pred_mask_b = pred_mask_b.astype(np.uint8)
            pred_mask_b = pred_mask_b[92:-92, 4:-4]
            pred_mask_b = remove_samll_object(pred_mask_b)
            masks.append(pred_mask_b.copy())
            pred_seg = seg_mutex(output_affs, offsets=offsets, strides=list(cfg.DATA.strides), mask=pred_mask_b).astype(np.uint16)
        pred_seg = merge_func(pred_seg)
        pred_seg = relabel(pred_seg)
        seg.append(pred_seg.copy())
        pred_seg = pred_seg.astype(np.uint16)
        gt_ins = gt_ins.astype(np.uint16)

        # evaluate
        temp_aji = agg_jc_index(gt_ins, pred_seg)
        temp_dice = pixel_f1(gt_ins, pred_seg)
        gt_relabel = remap_label(gt_ins, by_size=False)
        pred_relabel = remap_label(pred_seg, by_size=False)
        pq_info_cur = get_fast_pq(gt_relabel, pred_relabel, match_iou=0.5)[0]
        temp_f1 = pq_info_cur[0]
        temp_pq = pq_info_cur[2]
        print('image=%d, AJI=%.6f, Dice=%.6f, F1=%.6f, PQ=%.6f' % (k, temp_aji, temp_dice, temp_f1, temp_pq), flush=True)
        f_txt.write('image=%d, AJI=%.6f, Dice=%.6f, F1=%.6f, PQ=%.6f' % (k, temp_aji, temp_dice, temp_f1, temp_pq))
        f_txt.write('\n')
        aji_score.append(temp_aji)
        dice_score.append(temp_dice)
        f1_score.append(temp_f1)
        pq_score.append(temp_pq)
        if args.show:
            if args.norm_embedding:
                embedding = F.normalize(embedding, p=2, dim=1)
            if args.show_embedding:
                embedding = embedding[:, :, 92:-92, 4:-4]
                val_show_emd(k, output_affs[-args.show_num], embedding, pred_seg, gt_ins, affs_img_path)
            else:
                affs_gt = batch_data['affs'].numpy()[0,-args.show_num]
                affs_gt = affs_gt[92:-92, 4:-4]
                val_show(k, output_affs[-args.show_num], affs_gt, pred_seg, gt_ins, affs_img_path)
    cost_time = time.time() - start_time
    epoch_loss = sum(losses_valid) / len(losses_valid)
    aji_score = np.asarray(aji_score)
    dice_score = np.asarray(dice_score)
    f1_score = np.asarray(f1_score)
    pq_score = np.asarray(pq_score)
    mean_aji = np.mean(aji_score)
    std_aji = np.std(aji_score)
    mean_dice = np.mean(dice_score)
    std_dice = np.std(dice_score)
    mean_f1 = np.mean(f1_score)
    std_f1 = np.std(f1_score)
    mean_pq = np.mean(pq_score)
    std_pq = np.std(pq_score)
    print('model-%s, valid-loss=%.6f, AJI=%.6f(%.6f), Dice=%.6f(%.6f), F1=%.6f(%.6f), PQ=%.6f(%.6f)' % \
        (args.model_id, epoch_loss, mean_aji, std_aji, mean_dice, std_dice, mean_f1, std_f1, mean_pq, std_pq), flush=True)
    print('COST TIME = %.6f' % cost_time)

    f_txt.write('model-%s, valid-loss=%.6f, AJI=%.6f(%.6f), Dice=%.6f(%.6f), F1=%.6f(%.6f), PQ=%.6f(%.6f)' % \
        (args.model_id, epoch_loss, mean_aji, std_aji, mean_dice, std_dice, mean_f1, std_f1, mean_pq, std_pq))
    f_txt.write('\n')
    f_txt.close()

    # seg = np.asarray(seg, dtype=np.uint16)
    # f = h5py.File(os.path.join(out_affs, 'seg.hdf'), 'w')
    # f.create_dataset('main', data=seg, dtype=seg.dtype, compression='gzip')
    # f.close()

    if args.save:
        print('Save...')
        affs = np.asarray(affs, dtype=np.float32)
        masks = np.asarray(masks, dtype=np.float32)
        f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
        f.create_dataset('main', data=affs, dtype=affs.dtype, compression='gzip')
        f.create_dataset('mask', data=masks, dtype=masks.dtype, compression='gzip')
        f.close()

    print('Done')

