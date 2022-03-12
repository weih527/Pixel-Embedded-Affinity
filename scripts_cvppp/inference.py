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

from data.data_provider_deep import Provider, Validation
from utils.show import show_affs, val_show, val_show_emd
from utils.utils import setup_seed
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss, BCE_loss_func
# from utils.evaluate import BestDice, AbsDiffFGLabels, SymmetricBestDice
from lib.evaluate.CVPPP_evaluate import BestDice, AbsDiffFGLabels, SymmetricBestDice, SymmetricBestDice_max
from utils.seg_mutex import seg_mutex
from utils.lmc import multicut_multi
from utils.seg_waterz import seg_waterz
from utils.affinity_ours import multi_offset
from utils.postprocessing import merge_small_object
from utils.merge_small import merge_small_segments
from data.data_segmentation import relabel
from model.unet2d_residual import ResidualUNet2D_deep as ResidualUNet2D_affs
from loss.loss_embedding_mse import embedding_loss, embedding2affs

from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref

import warnings
warnings.filterwarnings("ignore")

def merge_func(seg, step=4):
    seg = merge_small_object(seg)
    seg = merge_small_object(seg, threshold=20, window=11)
    seg = merge_small_object(seg, threshold=50, window=11)
    # seg = merge_small_object(seg, threshold=100, window=21)
    seg = merge_small_object(seg, threshold=300, window=21)
    return seg

def merge_func2(seg, step=4):
    seg = merge_small_object(seg)
    seg = merge_small_object(seg, threshold=20, window=11)
    seg = merge_small_object(seg, threshold=50, window=11)
    # seg = merge_small_object(seg, threshold=100, window=21)
    seg = merge_small_object(seg, threshold=500, window=101)
    return seg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cvppp')
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
    parser.add_argument('-sbd', '--if_sbd', action='store_false', default=True)
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
    dice = []
    dice_max = []
    diff = []
    all_voi = []
    all_arand = []
    affs = []
    seg = []
    if args.stride is None:
        stride = list(cfg.DATA.strides)
    else:
        stride = [args.stride, args.stride]

    start_time = time.time()
    f_txt = open(os.path.join(out_affs, 'score.txt'), 'w')
    for k, batch in enumerate(val_loader, 0):
        # if k != 27:
        #     continue
        batch_data = batch
        inputs = batch_data['image'].cuda()
        target = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()
        target_ins = batch_data['seg'].cuda()
        affs_mask = batch_data['mask'].cuda()
        down1 = batch_data['down1'].cuda()
        down2 = batch_data['down2'].cuda()
        down3 = batch_data['down3'].cuda()
        down4 = batch_data['down4'].cuda()
        with torch.no_grad():
            emd4, emd3, emd2, emd1, embedding, pred_mask = model(inputs)
        if args.mode == 'test':
            losses_valid.append(0.0)
            pred = embedding2affs(embedding, offsets, mode=cfg.TRAIN.dis_mode)
        else:
            loss_emd1, _, _ = embedding_loss(emd1, down1[:,0:nb_half*4], down1[:,nb_half*4:nb_half*8], down1[:,nb_half*8:nb_half*12], criterion, offsets[:nb_half*4], affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
            loss_emd2, _, _ = embedding_loss(emd2, down2[:,0:nb_half*3], down2[:,nb_half*3:nb_half*6], down2[:,nb_half*6:nb_half*9], criterion, offsets[:nb_half*3], affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
            loss_emd3, _, _ = embedding_loss(emd3, down3[:,0:nb_half*2], down3[:,nb_half*2:nb_half*4], down3[:,nb_half*4:nb_half*6], criterion, offsets[:nb_half*2], affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
            loss_emd4, _, _ = embedding_loss(emd4, down4[:,0:nb_half*1], down4[:,nb_half*1:nb_half*2], down4[:,nb_half*2:nb_half*3], criterion, offsets[:nb_half*1], affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
            loss_embedding, pred, _ = embedding_loss(embedding, target, weightmap, affs_mask, criterion, offsets, affs0_weight=cfg.TRAIN.affs0_weight, mode=cfg.TRAIN.dis_mode)
            loss_mask = cfg.TRAIN.mask_weight * criterion_mask(pred_mask, torch.gt(target_ins[:, 0], 0), weight_rate=[10, 1]).to(device)
            tmp_loss = loss_emd1 + loss_emd2 + loss_emd3 + loss_emd4 + loss_embedding + loss_mask
            losses_valid.append(tmp_loss.item())
        pred = F.relu(pred)
        # pred = torch.abs(pred)
        output_affs = np.squeeze(pred.data.cpu().numpy())
        affs.append(output_affs.copy())

        # post-processing
        gt_ins = np.squeeze(batch_data['seg'].numpy()).astype(np.uint8)
        gt_mask = gt_ins.copy()
        gt_mask[gt_mask != 0] = 1
        pred_seg = seg_mutex(output_affs, offsets=offsets, strides=stride, mask=gt_mask).astype(np.uint16)
        # pred_seg, fragment = seg_waterz(output_affs[:2, ...], mask=gt_mask)
        # pred_seg = pred_seg.astype(np.uint16)
        # pred_seg = relabel(pred_seg) + 1
        # pred_seg = merge_small_segments(pred_seg, 100)
        # pred_seg[pred_seg == 1] = 0
        # pred_seg = multicut_multi(output_affs[:2]).astype(np.uint16)
        pred_seg = merge_func(pred_seg)
        pred_seg = relabel(pred_seg)
        seg.append(pred_seg.copy())
        pred_seg = pred_seg.astype(np.uint16)
        gt_ins = gt_ins.astype(np.uint16)

        # evaluate
        if args.mode == 'test':
            temp_dice = 0.0
            temp_diff = np.max(pred_seg)
            arand = 0.0
            voi_sum = 0.0
            temp_dice_max = 0.0
        else:
            if args.if_sbd:
                temp_dice = SymmetricBestDice(pred_seg, gt_ins)
                temp_dice_max = SymmetricBestDice_max(pred_seg, gt_ins)
            else:
                temp_dice = 0.0
                temp_dice_max = 0.0
            temp_diff = AbsDiffFGLabels(pred_seg, gt_ins)
            arand = adapted_rand_ref(gt_ins, pred_seg, ignore_labels=(0))[0]
            voi_split, voi_merge = voi_ref(gt_ins, pred_seg, ignore_labels=(0))
            voi_sum = voi_split + voi_merge
        print('image=%d, SBD=%.6f, DiC=%.6f, VOI=%.6f, ARAND=%.6f' % (k, temp_dice, temp_diff, voi_sum, arand))
        f_txt.write('image=%d, SBD=%.6f, DiC=%.6f, VOI=%.6f, ARAND=%.6f' % (k, temp_dice, temp_diff, voi_sum, arand))
        f_txt.write('\n')
        all_voi.append(voi_sum)
        all_arand.append(arand)
        dice.append(temp_dice)
        dice_max.append(temp_dice_max)
        diff.append(temp_diff)
        if args.show:
            if args.norm_embedding:
                embedding = F.normalize(embedding, p=2, dim=1)
            if args.mode == 'test':
                if args.show_embedding:
                    val_show_emd(k, output_affs[-args.show_num], embedding, pred_seg, gt_ins, affs_img_path)
                else:
                    val_show(k, output_affs[-args.show_num], output_affs[-args.show_num], pred_seg, gt_ins, affs_img_path)
            else:
                if args.show_embedding:
                    val_show_emd(k, output_affs[-args.show_num], embedding, pred_seg, gt_ins, affs_img_path)
                else:
                    affs_gt = batch_data['affs'].numpy()[0,-args.show_num]
                    val_show(k, output_affs[-args.show_num], affs_gt, pred_seg, gt_ins, affs_img_path)
    cost_time = time.time() - start_time
    epoch_loss = sum(losses_valid) / len(losses_valid)
    sbd = sum(dice) / len(dice)
    sbd_max = sum(dice_max) / len(dice_max)
    # sbd = 0.0
    dic = sum(diff) / len(diff)
    mean_voi = sum(all_voi) / len(all_voi)
    mean_arand = sum(all_arand) / len(all_arand)
    print('model-%s, valid-loss=%.6f, SBD_min=%.6f, SBD_max=%.6f, DiC=%.6f, VOI=%.6f, ARAND=%.6f' % \
        (args.model_id, epoch_loss, sbd, sbd_max, dic, mean_voi, mean_arand), flush=True)
    print('COST TIME = %.6f' % cost_time)

    f_txt.write('model-%s, valid-loss=%.6f, SBD_min=%.6f, SBD_max=%.6f, DiC=%.6f, VOI=%.6f, ARAND=%.6f' % \
        (args.model_id, epoch_loss, sbd, sbd_max, dic, mean_voi, mean_arand))
    f_txt.write('\n')
    f_txt.close()

    seg = np.asarray(seg, dtype=np.uint16)
    f = h5py.File(os.path.join(out_affs, 'seg.hdf'), 'w')
    f.create_dataset('main', data=seg, dtype=seg.dtype, compression='gzip')
    f.close()

    if args.mode == 'test':
        seg = seg[:, 7:-7, 22:-22]
        seg = seg.astype(np.uint8)
        out_seg_path = os.path.join(out_affs, 'submission.h5')
        example_path = '../data/CVPPP/submission_example.h5'
        copyfile(example_path, out_seg_path)
        fi = ['plant003','plant004','plant009','plant014','plant019','plant023','plant025','plant028','plant034',
            'plant041','plant056','plant066','plant074','plant075','plant081','plant087','plant093','plant095',
            'plant097','plant103','plant111','plant112','plant117','plant122','plant125','plant131','plant136',
            'plant140','plant150','plant155','plant157','plant158','plant160']
        f_out = h5py.File(out_seg_path, 'r+')
        for k, fn in enumerate(fi):
            data = f_out['A1']
            img = data[fn]['label'][:]
            del data[fn]['label']
            data[fn]['label'] = seg[k]
        f_out.close()

    if args.save:
        affs = np.asarray(affs, dtype=np.float32)
        f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
        f.create_dataset('main', data=affs, dtype=affs.dtype, compression='gzip')
        f.close()

