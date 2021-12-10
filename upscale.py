import argparse, os
import torch
from torch.autograd import Variable
# from scipy.ndimage import imread
from imageio import imread
from PIL import Image
import PIL
import numpy as np
import time, math
import matplotlib.pyplot as plt
import cv2
import ntpath



# default cuda true가 맞는지
parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
parser.add_argument("--cuda", default="true", action="store_true", help="use cuda?")
parser.add_argument("--model", default="upscale_model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="bird_GT", type=str, help="image name")
# parser.add_argument("--image", default="g1", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

def get_test_list(root_dir):
    test_list = os.listdir(root_dir)
    test_list = [os.path.join(root_dir, i) for i in test_list]
    return test_list


opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

# get image output list
img_transfer_list = get_test_list(os.getcwd()+'/img_transfer/')
img_transfer_list = [i for i in img_transfer_list if '.jpg' in i]
img_transfer_list.sort()


for m in range(len(img_transfer_list[:])):
    pname = os.path.splitext(img_transfer_list[m])[0]
    fname = ntpath.basename(pname)
    Image.open(pname+'.jpg').save(os.getcwd()+'/bmp_transfer/'+fname+'.bmp')



bmp_transfer_list = get_test_list(os.getcwd()+'/bmp_transfer/')
bmp_transfer_list = [i for i in bmp_transfer_list if '.bmp' in i]
bmp_transfer_list.sort()

upscale_output = os.getcwd()+'/upscale_output/'


for m in range(len(bmp_transfer_list[:])):
    pname = os.path.splitext(bmp_transfer_list[m])[0]
    fname = ntpath.basename(pname)

    im_b  = imread(bmp_transfer_list[m])
    im_b_ycbcr = cv2.cvtColor(im_b, cv2.COLOR_BGR2YCR_CB)
    im_b_y = im_b_ycbcr[:,:,0].astype(float)
    im_input = im_b_y/255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    out = model(im_input)
    elapsed_time = time.time() - start_time

    out = out.cpu()

    im_h_y = out.data[0].numpy().astype(np.float32)
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.

    im_h = colorize(im_h_y[0,:,:], im_b_ycbcr)
    im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")

    # print("It takes {}s for processing".format(elapsed_time))

    im_h.save(upscale_output + fname + ".jpg")
    #im_b.save(upscale_output + fname + ".jpg")
