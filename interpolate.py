from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time, math


def get_test_list(root_dir):
    test_list = os.listdir(root_dir)
    test_list = [os.path.join(root_dir, i) for i in test_list]
    return test_list

upscale_output_list = get_test_list(os.getcwd()+'/upscale_output/')
upscale_output_list = [i for i in upscale_output_list if '.jpg' in i]
upscale_output_list.sort()

final_output = os.getcwd()+'/final_output/'

NUM_OF_INTERPOLATE = 6
COUNT = 1

def interpolate(data1, data2, width, height,counter):
    for i in range(NUM_OF_INTERPOLATE):
        data = np.empty([width,height,3])
        for j in range(width):
            for k in range(height):
                v = i/NUM_OF_INTERPOLATE
                data[j][k] = (1-v)*data1[j][k]+v*data2[j][k]
        result = Image.fromarray(data.astype(np.uint8))
        result = result.save(final_output+str(counter)+".jpg")
        counter += 1



width = mpimg.imread(upscale_output_list[0]).shape[0]
height = mpimg.imread(upscale_output_list[0]).shape[1]

for i in range(4):
    anchor = 6*i
    for j in range(NUM_OF_INTERPOLATE):

        

        offset2 = anchor+j
        if j==0:
            offset1 = offset2+NUM_OF_INTERPOLATE-1
            img1 = mpimg.imread(upscale_output_list[offset1])
            data1 = np.array(img1)
        else:
            offset1 = offset2-1
            img1 = img2
            data1 = data2
        img2 = mpimg.imread(upscale_output_list[offset2])
        data2 = np.array(img2)
        2
        for q in range(NUM_OF_INTERPOLATE):
            v = q/NUM_OF_INTERPOLATE
            data = (1-v)*data1+v*data2
            result = Image.fromarray(data.astype(np.uint8))
            result = result.save(final_output+str(COUNT)+".jpg")
            #print(COUNT)
            COUNT += 1

