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

num = 80

seed = 1
np.random.seed(seed)
random.seed(seed)

path = './train'
files = os.listdir(path)
id_list = [f[:8] for f in files if 'rgb' in f]
print(len(id_list))

f_txt = open(os.path.join('./valid_set', 'local_20_1.txt'), 'r')
valid_set = [x[:-1] for x in f_txt.readlines()]
f_txt.close()
print(len(valid_set))

remain = remove_list(id_list, valid_set)
print(len(remain))

out = stride_sample(remain, num)

print(out)
f_txt = open(os.path.join('valid_set', 'remove_'+str(num)+'_'+str(seed)+'.txt'), 'w')
for k in out:
    f_txt.write(k)
    f_txt.write('\n')
f_txt.close()
