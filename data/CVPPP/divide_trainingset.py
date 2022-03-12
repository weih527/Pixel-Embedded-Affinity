'''
Descripttion: 
version: 0.0
Author: Wei Huang
Date: 2022-03-12 13:17:25
'''
import os
import random
import numpy as np

def stride_sample(samples, num):
    out = []
    sub_block = len(samples) // num
    remain = len(samples) % num
    for i in range(num):
        if i < remain:
            start = i * (sub_block + 1)
            end = (i+1) * (sub_block + 1)
        else:
            start = remain*(sub_block + 1) + (i-remain) * sub_block
            end = remain*(sub_block + 1) + (i-remain+1) * sub_block
        out += random.sample(samples[start:end], 1)
    return out

def remove_list(list1, list2):
    out = []
    for k in list1:
        if k in list2:
            continue
        out.append(k)
    return out

num = 20
mode = 'local' # global, local, cross

seed = 1
np.random.seed(seed)
random.seed(seed)

path = './train'
files = os.listdir(path)
id_list = [f[:8] for f in files if 'rgb' in f]
# print(id_list)

if mode == 'global':
    # globally sample 20 images used as validation set
    out = random.sample(id_list, num)

    print(out)
    f_txt = open(os.path.join('valid_set', mode+'_'+str(num)+'_'+str(seed)+'.txt'), 'w')
    for k in out:
        f_txt.write(k)
        f_txt.write('\n')
    f_txt.close()
elif mode == 'local':
    # locally sample 20 images used as validation set
    out = stride_sample(id_list, num)

    print(out)
    f_txt = open(os.path.join('valid_set', mode+'_'+str(num)+'_'+str(seed)+'.txt'), 'w')
    for k in out:
        f_txt.write(k)
        f_txt.write('\n')
    f_txt.close()
elif mode == 'cross':
    # 3-fold cross validation
    out1 = stride_sample(id_list, 43)
    print(out1)
    f_txt = open(os.path.join('valid_set', mode+'_1.txt'), 'w')
    for k in out1:
        f_txt.write(k)
        f_txt.write('\n')
    f_txt.close()

    remain = remove_list(id_list, out1)
    out2 = stride_sample(remain, 43)
    print(out2)
    f_txt = open(os.path.join('valid_set', mode+'_2.txt'), 'w')
    for k in out2:
        f_txt.write(k)
        f_txt.write('\n')
    f_txt.close()

    out3 = remove_list(remain, out2)
    print(out3)
    f_txt = open(os.path.join('valid_set', mode+'_3.txt'), 'w')
    for k in out3:
        f_txt.write(k)
        f_txt.write('\n')
    f_txt.close()
else:
    raise NotImplementedError
