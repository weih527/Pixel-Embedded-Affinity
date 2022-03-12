import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# show
def draw_fragments_2d(pred):
    m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("the number of instance is %d" % size)
    color_pred = np.zeros([m, n, 3], dtype=np.uint8)
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,i] = color_val[idx]
    return color_pred

def embedding_pca(embeddings, n_components=3, as_rgb=True):
    if as_rgb and n_components != 3:
        raise ValueError("")

    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype('uint8')
    return embed_flat

def show_raw_img(img):
    std = np.asarray([0.229, 0.224, 0.225])
    std = std[np.newaxis, np.newaxis, :]
    mean = np.asarray([0.485, 0.456, 0.406])
    mean = mean[np.newaxis, np.newaxis, :]
    if img.shape[0] == 3:
        img = np.transpose(img, (1,2,0))
    img = ((img * std + mean) * 255).astype(np.uint8)
    return img

def show_affs(iters, inputs, pred, target, cache_path, if_cuda=False):
    pred = pred[0].data.cpu().numpy()
    if if_cuda:
        inputs = inputs[0].data.cpu().numpy()
        target = target[0].data.cpu().numpy()
    else:
        inputs = inputs[0].numpy()
        target = target[0].numpy()
    inputs = show_raw_img(inputs)
    pred[pred<0]=0; pred[pred>1]=1
    target[target<0]=0; target[target>1]=1
    pred = (pred * 255).astype(np.uint8)
    pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    target = (target * 255).astype(np.uint8)
    target = np.repeat(target[:,:,np.newaxis], 3, 2)
    im_cat = np.concatenate([inputs, pred, target], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def show_affs_emb(iters, inputs, ema_inputs, pred, target, emb1, emb2, cache_path, if_cuda=False):
    pred = pred[0].data.cpu().numpy()
    if if_cuda:
        inputs = inputs[0].data.cpu().numpy()
        ema_inputs = ema_inputs[0].data.cpu().numpy()
        target = target[0].data.cpu().numpy()
    else:
        inputs = inputs[0].numpy()
        ema_inputs = ema_inputs[0].numpy()
        target = target[0].numpy()
    emb1 = emb1[0].data.cpu().numpy()
    emb1 = embedding_pca(emb1)
    emb1 = np.transpose(emb1, (1,2,0))
    emb2 = emb2[0].data.cpu().numpy()
    emb2 = embedding_pca(emb2)
    emb2 = np.transpose(emb2, (1,2,0))
    inputs = show_raw_img(inputs)
    ema_inputs = show_raw_img(ema_inputs)
    pred[pred<0]=0; pred[pred>1]=1
    target[target<0]=0; target[target>1]=1
    pred = (pred * 255).astype(np.uint8)
    pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    target = (target * 255).astype(np.uint8)
    target = np.repeat(target[:,:,np.newaxis], 3, 2)
    im_cat1 = np.concatenate([inputs, emb1, pred], axis=1)
    im_cat2 = np.concatenate([ema_inputs, emb2, target], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def val_show(iters, pred, target, pred_seg, gt_ins, valid_path):
    pred[pred<0]=0; pred[pred>1]=1
    target[target<0]=0; target[target>1]=1
    pred = pred[:,:,np.newaxis]
    pred = np.repeat(pred, 3, 2)
    target = target[:,:,np.newaxis]
    target = np.repeat(target, 3, 2)
    pred = (pred * 255).astype(np.uint8)
    target = (target * 255).astype(np.uint8)
    im_cat1 = np.concatenate([pred, target], axis=1)
    seg_color = draw_fragments_2d(pred_seg)
    ins_color = draw_fragments_2d(gt_ins)
    im_cat2 = np.concatenate([seg_color, ins_color], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(valid_path, '%06d.png' % iters))

def val_show_emd(iters, pred, embedding, pred_seg, gt_ins, valid_path):
    pred[pred<0]=0; pred[pred>1]=1
    embedding = np.squeeze(embedding.data.cpu().numpy())
    embedding = embedding_pca(embedding)
    embedding = np.transpose(embedding, (1,2,0))
    pred = pred[:,:,np.newaxis]
    pred = np.repeat(pred, 3, 2)
    pred = (pred * 255).astype(np.uint8)
    im_cat1 = np.concatenate([pred, embedding], axis=1)
    seg_color = draw_fragments_2d(pred_seg)
    ins_color = draw_fragments_2d(gt_ins)
    im_cat2 = np.concatenate([seg_color, ins_color], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(valid_path, '%06d.png' % iters))
