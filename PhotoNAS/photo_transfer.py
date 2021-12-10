import os
import cv2
import sys
import torch
import ntpath
import argparse
import torchfile
import time, math
import numpy as np
import torch.nn as nn
from wct import transform
from dataset import TransferDataset
from matplotlib import pyplot as plt
from torchvision import utils, transforms


# please change directory to your own environments
#main_path = '/home/junhyub/documents/timetraveler_module/'
main_path = os.getcwd()+'/'

sys.path.append(main_path)
abs_dir = os.path.abspath(os.path.dirname(__file__))

from models.models_photorealistic_nas.VGG_with_decoder import encoder, decoder0, decoder1, decoder2, decoder3, decoder4, decoder5



def load_net():
    # get vgg parameters
    encoder_param = torchfile.load(main_path+'models/models_photorealistic_nas/vgg_normalised_conv5_1.t7')

    # train model directory list
    #################################################
    models_directory = 'trained_models_aaai_addtrain'
    #models_directory = 'trained_models_aaai_simple'
    #################################################

    # net settings
    net_e = encoder(encoder_param)
    net_d0 = decoder0()
    net_d0.load_state_dict(torch.load(os.path.join(abs_dir, models_directory + '/decoder_epoch_2.pth.tar')))
    net_d1 = decoder1()
    net_d1.load_state_dict(torch.load(os.path.join(abs_dir, models_directory + '/decoder_epoch_2.pth.tar')))
    net_d2 = decoder2()
    net_d2.load_state_dict(torch.load(os.path.join(abs_dir, models_directory + '/decoder_epoch_2.pth.tar')))
    net_d3 = decoder3()
    net_d3.load_state_dict(torch.load(os.path.join(abs_dir, models_directory + '/decoder_epoch_2.pth.tar')))
    net_d4 = decoder4()
    net_d4.load_state_dict(torch.load(os.path.join(abs_dir, models_directory + '/decoder_epoch_2.pth.tar')))
    net_d5 = decoder5()
    net_d5.load_state_dict(torch.load(os.path.join(abs_dir, models_directory + '/decoder_epoch_2.pth.tar')))
    return net_e, net_d0, net_d1, net_d2, net_d3, net_d4, net_d5

def get_test_list(root_dir):
    test_list = os.listdir(root_dir)
    test_list = [os.path.join(root_dir, i) for i in test_list]
    return test_list

def get_a_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def resize_save(content, style, out):
    if content.shape[0] < content.shape[1]:
        out_h = 512
        out_w = np.int32(512.0 * content.shape[1] / content.shape[0])
    else:
        out_w = 512
        out_h = np.int32(512.0 * content.shape[0] / content.shape[1])
    content = cv2.resize(content, (out_w, out_h), cv2.INTER_AREA)
    style = cv2.resize(style, (out_w, out_h), cv2.INTER_AREA)
    out = cv2.resize(out, (out_w, out_h), cv2.INTER_AREA)
    return content, style, out

def resize_imgs(content, style):
    c_h = 512
    c_w = 768
    s_h = 512
    s_w = 768
    content = cv2.resize(content, (c_w, c_h), cv2.INTER_AREA)
    style   = cv2.resize(style,   (s_w, s_h), cv2.INTER_AREA)
    return content, style

if __name__ == '__main__':

    # argument parser settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=1)
    parser.add_argument('-sd', '--save_dir', default=main_path+'img_transfer/')
    parser.add_argument('-c', '--content', default=main_path+'img_input/')
    parser.add_argument('-s', '--style', default=main_path+'img_style/')
    parser.add_argument('-a', '--alpha', default=1.0)
    parser.add_argument('-d', '--d_control', default='01010000000100000000000000001111')


    args = parser.parse_args()
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    net_e, net_d0, net_d1, net_d2, net_d3, net_d4, net_d5 = load_net()
    d0_control = args.d_control[:5]
    d1_control = args.d_control[5: 8]
    d2_control = args.d_control[9: 16]
    d3_control = args.d_control[16: 23]
    d4_control = args.d_control[23: 28]
    d5_control = args.d_control[28: 32]
    d0_control = [int(i) for i in d0_control]
    d1_control = [int(i) for i in d1_control]
    d2_control = [int(i) for i in d2_control]
    d3_control = [int(i) for i in d3_control]
    d4_control = [int(i) for i in d4_control]
    d5_control = [int(i) for i in d5_control]

    if args.gpu is not None:
        net_e.cuda(), net_e.eval() 
        net_d0.cuda(), net_d0.eval()
        net_d1.cuda(), net_d1.eval()
        net_d2.cuda(), net_d2.eval()
        net_d3.cuda(), net_d3.eval()
        net_d4.cuda(), net_d4.eval()
        net_d5.cuda(), net_d5.eval()

    # get content image list
    content_list = get_test_list(args.content)
    content_list = [i for i in content_list if '.jpg' in i]
    content_list.sort()

    # get style image list
    style_list = get_test_list(args.style)
    style_list = [i for i in style_list if '.jpg' in i]
    style_list.sort()


    for m in range(len(content_list[:])):
        for k in range(len(style_list[:])):
            # print('------- transfering pair content : {} | style : {} -------'.format(m,k))

            # get content image
            content_path = content_list[m]
            content = get_a_image(content_path)   

            # get content image
            style_path = style_list[k]
            style = get_a_image(style_path)   

            # force resize -> should be auto resize later
            content, style = resize_imgs(content, style)

            content = transforms.ToTensor()(content)
            content = content.unsqueeze(0)

            style   = transforms.ToTensor()(style)
            style   = style.unsqueeze(0)

            if args.gpu is not None:
                content = content.cuda()
                style   = style.cuda()

            cF = list(net_e(content))
            sF = list(net_e(style))
            
            csF = []
            for ii in range(len(cF)):
                if ii == 0:
                    if d0_control[0] == 1:
                        this_csF = transform(cF[ii], sF[ii], args.alpha)
                        csF.append(this_csF)
                    else:
                        csF.append(cF[ii])
                elif ii == 1:
                    if d2_control[-1] == 1:
                        this_csF = transform(cF[ii], sF[ii], args.alpha)
                        csF.append(this_csF)
                    else:
                        csF.append(cF[ii])
                elif ii == 2:
                    if d3_control[-1] == 1:
                        this_csF = transform(cF[ii], sF[ii], args.alpha)
                        csF.append(this_csF)
                    else:
                        csF.append(cF[ii])
                elif ii == 3:
                    if d4_control[-1] == 1:
                        this_csF = transform(cF[ii], sF[ii], args.alpha)
                        csF.append(this_csF)
                    else:
                        csF.append(cF[ii])
                elif ii == 4:
                    if d5_control[-1] == 1:
                        this_csF = transform(cF[ii], sF[ii], args.alpha)
                        csF.append(this_csF)
                    else:
                        csF.append(cF[ii])
                else:
                    csF.append(cF[ii])

            csF[0] = net_d0(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            sF[0]  = net_d0( *sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)

            if d1_control[0] == 1:
                csF[0] = transform(csF[0], sF[0], args.alpha)
            csF[0] = net_d1(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            sF[0]  = net_d1( *sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            if d2_control[0] == 1:
                csF[0] = transform(csF[0], sF[0], args.alpha)
            csF[0] = net_d2(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            sF[0]  = net_d2( *sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            if d3_control[0] == 1:
                csF[0] = transform(csF[0], sF[0], args.alpha)
            csF[0] = net_d3(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            sF[0]  = net_d3( *sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            if d4_control[0] == 1:
                csF[0] = transform(csF[0], sF[0], args.alpha)
            csF[0] = net_d4(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            sF[0]  = net_d4( *sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            if d5_control[0] == 1:
                csF[0] = transform(csF[0], sF[0], args.alpha)
            csF[0] = net_d5(*csF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)
            sF[0]  = net_d5( *sF, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control)


            # save result images
            pname = os.path.splitext(style_list[k])[0]
            fname = ntpath.basename(pname)
            out = csF[0].cpu().data.float()
            utils.save_image(out, os.path.join(args.save_dir, fname+'.jpg'))
            out = cv2.imread(os.path.join(args.save_dir, fname+'.jpg'))
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(args.save_dir, fname+'.jpg'), out)
            # utils.save_image(out, os.path.join(args.save_dir, '%d.jpg' % (k)))
            # out = cv2.imread(os.path.join(args.save_dir, '%d.jpg' % (k)))
            # out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(os.path.join(args.save_dir, '%d.jpg' % (k)), out)    
    

