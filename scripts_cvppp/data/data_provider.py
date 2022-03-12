import os
import cv2
import sys
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset.transforms import RandomAffine
from dataset.data_aug import scale, flip_crop
from utils.affinity_ours import multi_offset, gen_affs_ours
from data.data_segmentation import seg_widen_border, weight_binary_ratio
from data.data_consistency import Filp_EMA
from utils.utils import remove_list
from utils.consistency_aug import tensor2img, img2tensor, add_gauss_noise
from utils.consistency_aug import add_gauss_blur, add_intensity, add_mask

class ToLogits(object):
    def __init__(self, expand_dim=None):
        self.expand_dim = expand_dim

    def __call__(self, pic):
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int32, copy=True))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if self.expand_dim is not None:
            return img.unsqueeze(self.expand_dim)
        return img


class Train(Dataset):
    def __init__(self, cfg, mode='train'):
        super(Train, self).__init__()
        self.size = cfg.DATA.size
        self.flip = True
        self.crop = True
        self.aug_mode = cfg.DATA.aug_mode
        self.if_ema_flip = cfg.DATA.if_ema_flip
        self.if_ema_noise = cfg.DATA.if_ema_noise
        self.if_ema_blur = cfg.DATA.if_ema_blur
        self.if_ema_intensity = cfg.DATA.if_ema_intensity
        self.if_ema_mask = cfg.DATA.if_ema_mask
        print('augmentation mode:', self.aug_mode)
        self.data_folder = cfg.DATA.data_folder
        self.mode = mode
        self.padding = cfg.DATA.padding
        self.num_train = cfg.DATA.num_train
        self.separate_weight = cfg.DATA.separate_weight
        self.offsets = multi_offset(list(cfg.DATA.shifts), neighbor=cfg.DATA.neighbor)
        self.nb_half = cfg.DATA.neighbor // 2
        if (self.mode != "train") and (self.mode != "validation") and (self.mode != "test"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")
        if self.mode == "validation":
            self.dir = os.path.join(self.data_folder, "train")
        else:
            self.dir = os.path.join(self.data_folder, mode)

        self.id_num = os.listdir(self.dir)  # all file
        if self.mode == "test":
            self.id_img = [f for f in self.id_num if 'rgb' in f]
            self.id_label = [f for f in self.id_num if 'label' in f]
            self.id_fg = [f for f in self.id_num if 'fg' in f]

            self.id_img.sort(key=lambda x: int(x[5:8]))
            self.id_label.sort(key=lambda x: int(x[5:8]))
            self.id_fg.sort(key=lambda x: int(x[5:8]))
        else:
            print('valid set: ' + cfg.DATA.valid_set)
            f_txt = open(os.path.join(self.data_folder, "valid_set", cfg.DATA.valid_set+'.txt'), 'r')
            valid_set = [x[:-1] for x in f_txt.readlines()]
            f_txt.close()

            if self.mode == "validation":
                self.id_img = [x+'_rgb.png' for x in valid_set]
                self.id_label = [x+'_label.png' for x in valid_set]
                self.id_fg = [x+'_fg.png' for x in valid_set]
            if self.mode == "train":
                all_set = [f[:8] for f in self.id_num if 'rgb' in f]
                
                # if cfg.TRAIN.if_valid:
                #     train_set = remove_list(all_set, valid_set)
                # else:
                #     train_set = all_set
                if cfg.MODEL.finetuning:
                    train_set = all_set
                else:
                    train_set = remove_list(all_set, valid_set)
                
                if cfg.DATA.remove_training_set is not None:
                    print('remove training set: ' + cfg.DATA.remove_training_set)
                    f_txt = open(os.path.join(self.data_folder, "valid_set", cfg.DATA.remove_training_set+'.txt'), 'r')
                    remove_set = [x[:-1] for x in f_txt.readlines()]
                    f_txt.close()
                    train_set = remove_list(train_set, remove_set)
                
                self.id_img = [x+'_rgb.png' for x in train_set]
                self.id_label = [x+'_label.png' for x in train_set]
                self.id_fg = [x+'_fg.png' for x in train_set]
        print('The number of %s image is %d' % (self.mode, len(self.id_img)))

        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomResizedCrop(self.size, scale=(0.7, 1.)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.target_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomResizedCrop(self.size, scale=(0.7, 1.), interpolation=0),
             ToLogits()])

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.target_transform_test = transforms.Compose(
             [ToLogits()])

        rotation = 10
        translation = 0.1
        shear = 0.1
        zoom_factor = 0.7
        self.augmentation_transform = RandomAffine(rotation_range=rotation,
                                                translation_range=translation,
                                                shear_range=shear,
                                                zoom_range=(zoom_factor,1),
                                                interp='nearest')
        # self.augmentation_transform = RandomAffine(rotation_range=rotation,
        #                                         translation_range=translation,
        #                                         shear_range=shear,
        #                                         interp='nearest')

        self.ema_flip = Filp_EMA()

    def __getitem__(self, idx):
        k = random.randint(0, len(self.id_img)-1)
        if self.aug_mode == 'xiaoyu':
            data = Image.open(os.path.join(self.dir, self.id_img[k])).convert('RGB')
            label = Image.open(os.path.join(self.dir, self.id_label[k]))

            if self.padding:
                data = np.asarray(data)
                data = np.pad(data, ((7,7),(22,22),(0,0)), mode='reflect')
                data = Image.fromarray(data)
                label = np.asarray(label)
                label = np.pad(label, ((7,7),(22,22)), mode='constant')
                label = Image.fromarray(label)

            seed = np.random.randint(2147483647)
            random.seed(seed)
            data = self.transform(data)
            random.seed(seed)
            label = self.target_transform(label)
        else:
            data = Image.open(os.path.join(self.dir, self.id_img[k])).convert('RGB')
            label = np.asarray(Image.open(os.path.join(self.dir, self.id_label[k])))

            if self.padding:
                data = np.asarray(data)
                data = np.pad(data, ((7,7),(22,22),(0,0)), mode='reflect')
                data = Image.fromarray(data)
                label = np.pad(label, ((7,7),(22,22)), mode='constant')
            else:
                image_resize = transforms.Resize((self.size, self.size), Image.BILINEAR)
                data = image_resize(data)
            mask = np.zeros_like(label)
            mask[label != 0] = 1

            data = self.transform_test(data)
            label, mask = scale(data, label, mask)
            data, label, mask = flip_crop(data, label, mask, flip=self.flip, crop=self.crop, imsize=self.size)
            label = label.float()
            mask = mask.float()
            if np.random.rand() < 0.5:
                data, label, mask = self.augmentation_transform(data, label, mask)

        label_numpy = np.squeeze(label.numpy())
        label_down1 = cv2.resize(label_numpy, (0,0), fx=1/2, fy=1/2, interpolation=cv2.INTER_NEAREST)
        label_down2 = cv2.resize(label_numpy, (0,0), fx=1/4, fy=1/4, interpolation=cv2.INTER_NEAREST)
        label_down3 = cv2.resize(label_numpy, (0,0), fx=1/8, fy=1/8, interpolation=cv2.INTER_NEAREST)
        label_down4 = cv2.resize(label_numpy, (0,0), fx=1/16, fy=1/16, interpolation=cv2.INTER_NEAREST)
        lb_affs, affs_mask = gen_affs_ours(label_numpy, offsets=self.offsets, ignore=False, padding=True)
        lb_affs1, affs_mask1 = gen_affs_ours(label_down1, offsets=self.offsets[:self.nb_half*4], ignore=False, padding=True)
        lb_affs2, affs_mask2 = gen_affs_ours(label_down2, offsets=self.offsets[:self.nb_half*3], ignore=False, padding=True)
        lb_affs3, affs_mask3 = gen_affs_ours(label_down3, offsets=self.offsets[:self.nb_half*2], ignore=False, padding=True)
        lb_affs4, affs_mask4 = gen_affs_ours(label_down4, offsets=self.offsets[:self.nb_half*1], ignore=False, padding=True)
        # lb_affs, affs_mask = gen_affs_ours(label_numpy, offsets=self.offsets, ignore=False, padding=True)
        if self.separate_weight:
            weightmap = np.zeros_like(lb_affs)
            weightmap1 = np.zeros_like(lb_affs1)
            weightmap2 = np.zeros_like(lb_affs2)
            weightmap3 = np.zeros_like(lb_affs3)
            weightmap4 = np.zeros_like(lb_affs4)
            for i in range(lb_affs.shape[0]):
                weightmap[i] = weight_binary_ratio(lb_affs[i])
            for i in range(lb_affs1.shape[0]):
                weightmap1[i] = weight_binary_ratio(lb_affs1[i])
            for i in range(lb_affs2.shape[0]):
                weightmap2[i] = weight_binary_ratio(lb_affs2[i])
            for i in range(lb_affs3.shape[0]):
                weightmap3[i] = weight_binary_ratio(lb_affs3[i])
            for i in range(lb_affs4.shape[0]):
                weightmap4[i] = weight_binary_ratio(lb_affs4[i])
        else:
            weightmap = weight_binary_ratio(lb_affs)
            weightmap1 = weight_binary_ratio(lb_affs1)
            weightmap2 = weight_binary_ratio(lb_affs2)
            weightmap3 = weight_binary_ratio(lb_affs3)
            weightmap4 = weight_binary_ratio(lb_affs4)

        lb_affs = torch.from_numpy(lb_affs)
        weightmap = torch.from_numpy(weightmap)
        affs_mask = torch.from_numpy(affs_mask)
        down1 = torch.from_numpy(np.concatenate([lb_affs1, weightmap1, affs_mask1], axis=0))
        down2 = torch.from_numpy(np.concatenate([lb_affs2, weightmap2, affs_mask2], axis=0))
        down3 = torch.from_numpy(np.concatenate([lb_affs3, weightmap3, affs_mask3], axis=0))
        down4 = torch.from_numpy(np.concatenate([lb_affs4, weightmap4, affs_mask4], axis=0))

        ema_data = tensor2img(data).copy()
        if self.if_ema_noise:
            ema_data = add_gauss_noise(ema_data)

        if self.if_ema_blur:
            ema_data = add_gauss_blur(ema_data)

        if self.if_ema_intensity:
            ema_data = add_intensity(ema_data)

        if self.if_ema_mask:
            label_mask = label_numpy.copy()
            label_mask[label_mask != 0] = 1
            ema_data = add_mask(ema_data, label_mask)

        ema_data = img2tensor(ema_data)
        if self.if_ema_flip:
            ema_data, rule = self.ema_flip(ema_data)
            rule = torch.from_numpy(rule.astype(np.float32))
        else:
            rule = torch.from_numpy(np.asarray([0,0,0], dtype=np.float32))

        return {'image': data,
                'affs': lb_affs,
                'wmap': weightmap,
                'seg': label,
                'mask': affs_mask,
                'down1': down1,
                'down2': down2,
                'down3': down3,
                'down4': down4,
                'ema_image': ema_data,
                'rules': rule}

    def __len__(self):
        # return len(self.id_img)
        return int(sys.maxsize)


class Validation(Train):
    def __init__(self, cfg, mode='validation'):
        super(Validation, self).__init__(cfg, mode)
        self.mode = mode

    def __getitem__(self, k):
        data = Image.open(os.path.join(self.dir, self.id_img[k])).convert('RGB')
        if self.mode == 'test':
            label = Image.open(os.path.join(self.dir, self.id_fg[k]))
        else:
            label = Image.open(os.path.join(self.dir, self.id_label[k]))

        # if self.padding:
        data = np.asarray(data)
        data = np.pad(data, ((7,7),(22,22),(0,0)), mode='reflect')
        data = Image.fromarray(data)
        label = np.asarray(label)
        label = np.pad(label, ((7,7),(22,22)), mode='constant')
        label = Image.fromarray(label)

        data = self.transform_test(data)
        label = self.target_transform_test(label)

        rule = torch.from_numpy(np.asarray([0,0,0], dtype=np.float32))

        if self.mode == 'test':
            return {'image': data,
                'affs': data,
                'wmap': data,
                'seg': label,
                'mask': label,
                'down1': label,
                'down2': label,
                'down3': label,
                'down4': label,
                'ema_image': data,
                'rules': rule}
        else:
            label_numpy = np.squeeze(label.numpy())
            lb_affs, affs_mask = gen_affs_ours(label_numpy, offsets=self.offsets, ignore=False, padding=True)
            if self.separate_weight:
                weightmap = np.zeros_like(lb_affs)
                for i in range(len(self.offsets)):
                    weightmap[i] = weight_binary_ratio(lb_affs[i])
            else:
                weightmap = weight_binary_ratio(lb_affs)

            lb_affs = torch.from_numpy(lb_affs)
            weightmap = torch.from_numpy(weightmap)
            affs_mask = torch.from_numpy(affs_mask)
            return {'image': data,
                    'affs': lb_affs,
                    'wmap': weightmap,
                    'seg': label,
                    'mask': affs_mask,
                    'down1': label,
                    'down2': label,
                    'down3': label,
                    'down4': label,
                    'ema_image': data,
                    'rules': rule}

    def __len__(self):
        return len(self.id_img)

def collate_fn(batchs):
    batch_imgs = []
    batch_affs = []
    batch_wmap = []
    batch_seg = []
    batch_mask = []
    batch_down1 = []
    batch_down2 = []
    batch_down3 = []
    batch_down4 = []
    batch_ema_data = []
    batch_rules = []
    for batch in batchs:
        batch_imgs.append(batch['image'])
        batch_affs.append(batch['affs'])
        batch_wmap.append(batch['wmap'])
        batch_seg.append(batch['seg'])
        batch_mask.append(batch['mask'])
        batch_down1.append(batch['down1'])
        batch_down2.append(batch['down2'])
        batch_down3.append(batch['down3'])
        batch_down4.append(batch['down4'])
        batch_ema_data.append(batch['ema_image'])
        batch_rules.append(batch['rules'])
    
    batch_imgs = torch.stack(batch_imgs, 0)
    batch_affs = torch.stack(batch_affs, 0)
    batch_wmap = torch.stack(batch_wmap, 0)
    batch_seg = torch.stack(batch_seg, 0)
    batch_mask = torch.stack(batch_mask, 0)
    batch_down1 = torch.stack(batch_down1, 0)
    batch_down2 = torch.stack(batch_down2, 0)
    batch_down3 = torch.stack(batch_down3, 0)
    batch_down4 = torch.stack(batch_down4, 0)
    batch_ema_data = torch.stack(batch_ema_data, 0)
    batch_rules = torch.stack(batch_rules, 0)
    return {'image':batch_imgs,
            'affs': batch_affs,
            'wmap': batch_wmap,
            'seg': batch_seg,
            'mask': batch_mask,
            'down1': batch_down1,
            'down2': batch_down2,
            'down3': batch_down3,
            'down4': batch_down4,
            'ema_image': batch_ema_data,
            'rules': batch_rules}

class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = Train(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        # return self.data.num_per_epoch
        return int(sys.maxsize)

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                            shuffle=False, collate_fn=collate_fn, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                            shuffle=False, collate_fn=collate_fn, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            return batch


def show_batch(temp_data, out_path):
    tmp_data = temp_data['image']
    affs = temp_data['affs']
    weightmap = temp_data['wmap']
    seg = temp_data['seg']
    ema_data = temp_data['ema_image']
    rules = temp_data['rules']

    tmp_data = tmp_data.numpy()
    tmp_data = show_raw_img(tmp_data)

    ema_verse = simple_augment_reverse_torch(ema_data, rules.numpy().astype(np.uint8))
    ema_data = ema_data.numpy()
    ema_data = show_raw_img(ema_data)

    ema_verse = ema_verse.numpy()
    ema_verse = show_raw_img(ema_verse)

    shift = -5
    seg = np.squeeze(seg.numpy().astype(np.uint8))
    # seg = seg[shift]
    # seg = seg[:,:,np.newaxis]
    # seg = np.repeat(seg, 3, 2)
    # seg_color = (seg * 255).astype(np.uint8)
    seg_color = draw_fragments_2d(seg)

    # affs = np.squeeze(affs.numpy())
    # affs = affs[shift]
    # affs = affs[:,:,np.newaxis]
    # affs = np.repeat(affs, 3, 2)
    # affs = (affs * 255).astype(np.uint8)

    im_cat = np.concatenate([tmp_data, seg_color, ema_data], axis=1)
    Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))

if __name__ == "__main__":
    import yaml
    from attrdict import AttrDict
    from utils.show import show_raw_img, draw_fragments_2d
    from data.data_consistency import simple_augment_reverse_torch
    seed = 555
    np.random.seed(seed)
    random.seed(seed)

    cfg_file = 'cvppp_ct_deep2_embedding_mse_l201_iy_mk_dw1.yaml'
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))
    
    out_path = os.path.join('./', 'data_temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    data = Train(cfg)
    for i in range(0, 50):
        temp_data = iter(data).__next__()
        show_batch(temp_data, out_path)

    # data = Validation(cfg, mode='validation')
    # for i, temp_data in enumerate(data):
    #     show_batch(temp_data, out_path)
