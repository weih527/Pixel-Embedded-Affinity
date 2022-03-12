'''
Descripttion: 
version: 0.0
Author: Wei Huang
Date: 2022-03-12 15:19:04
'''
import os
import numpy as np
from PIL import Image
import skimage.io
import skimage.segmentation

input_path = '../data/BBBC039V1/masks'
file_list = os.listdir(input_path)
output_path = '../data/BBBC039V1/label_instance'
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_path2 = '../data/BBBC039V1/label_boundary'
if not os.path.exists(output_path2):
    os.makedirs(output_path2)

total_objects = 0
for filename in file_list:
    if filename[-4:] == '.png':
        print(filename)
        annot = skimage.io.imread(os.path.join(input_path, filename))

        # strip the first channel
        if len(annot.shape) == 3:
            annot = annot[:,:,0]
        
        # label the annotations nicely to prepare for future filtering operation
        annot = skimage.morphology.label(annot)
        total_objects += len(np.unique(annot)) - 1
        
        # filter small objects, e.g. micronulcei
        annot = skimage.morphology.remove_small_objects(annot, min_size=25)
        assert annot.max() < 256, "the id of instance must be smaller than 256"
        annot = annot.astype(np.uint8)
        Image.fromarray(annot).save(os.path.join(output_path, filename))
        
        # find boundaries
        boundaries = skimage.segmentation.find_boundaries(annot)

        for k in range(2, 2, 2):
            boundaries = skimage.morphology.binary_dilation(boundaries)
            
        # BINARY LABEL
        
        # prepare buffer for binary label
        label_binary = np.zeros((annot.shape + (3,)))
        
        # write binary label
        label_binary[(annot == 0) & (boundaries == 0), 0] = 1
        label_binary[(annot != 0) & (boundaries == 0), 1] = 1
        label_binary[boundaries == 1, 2] = 1
        
        # save it - converts image to range from 0 to 255
        skimage.io.imsave(os.path.join(output_path2, filename), label_binary)
print("Total objects: ",total_objects)
