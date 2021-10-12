import torch.nn as nn
import torch
import numpy as np
# from IPython import embed

class encoder(nn.Module):
    def __init__(self,vgg):
        super(encoder,self).__init__()
        # vgg
        # 224 x 224

        # vgg.shape
        # actually not using
        # a = vgg._obj.modules
        # b = vgg._obj.modules[0].weight
        # c = vgg._obj.modules[0].weight
        # t = torch.Tensor(vgg._obj.modules[0].weight)
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.conv1.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[0].weight))
        self.conv1.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[0].bias))
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.conv2.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[2].weight))
        self.conv2.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[2].bias))
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.conv3.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[5].weight))
        self.conv3.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[5].bias))
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.conv4.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[9].weight))
        self.conv4.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[9].bias))
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.conv5.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[12].weight))
        self.conv5.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[12].bias))
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.conv6.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[16].weight))
        self.conv6.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[16].bias))
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,256,3,1,0)
        self.conv7.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[19].weight))
        self.conv7.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[19].bias))
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(256,256,3,1,0)
        self.conv8.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[22].weight))
        self.conv8.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[22].bias))
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(256,256,3,1,0)
        self.conv9.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[25].weight))
        self.conv9.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[25].bias))
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(256,512,3,1,0)
        self.conv10.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[29].weight))
        self.conv10.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[29].bias))
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,512,3,1,0)
        self.conv11.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[32].weight))
        self.conv11.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[32].bias))
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(512,512,3,1,0)
        self.conv12.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[35].weight))
        self.conv12.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[35].bias))
        self.relu12 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(512,512,3,1,0)
        self.conv13.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[38].weight))
        self.conv13.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[38].bias))
        self.relu13 = nn.ReLU(inplace=True)
        # 28 x 28

        self.maxPool4 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 14 x 14

        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(512,512,3,1,0)
        self.conv14.weight = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[42].weight))
        self.conv14.bias = torch.nn.Parameter(torch.Tensor(vgg._obj.modules[42].bias))
        self.relu14 = nn.ReLU(inplace=True)
        # 14 x 14
    def forward(self,x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out1 = out
        out = self.reflecPad3(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out,pool_idx = self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out2 = out
        out = self.reflecPad5(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out,pool_idx2 = self.maxPool2(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out3 = out
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out,pool_idx3 = self.maxPool3(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out4 = out
        out = self.reflecPad11(out)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out,pool_idx4 = self.maxPool4(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        return out, out4, out3, out2, out1

class decoder(nn.Module):
    def __init__(self,d=None):
        super(decoder,self).__init__()
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=4,stride=4,return_indices = True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=8,stride=8,return_indices = True)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=16,stride=16,return_indices = True)
        self.nn01 = nn.InstanceNorm2d(512)
        self.nn02 = nn.InstanceNorm2d(256)
        self.nn03 = nn.InstanceNorm2d(128)
        self.nn04 = nn.InstanceNorm2d(64)

        self.conv_pyramid0 = nn.Conv2d(512+0,512,1,1,0)
        self.conv_pyramid11 = nn.Conv2d(512+512,512,1,1,0)
        self.conv_pyramid12 = nn.Conv2d(512+256,512,1,1,0)
        self.conv_pyramid13 = nn.Conv2d(512+128,512,1,1,0)
        self.conv_pyramid14 = nn.Conv2d(512+64,512,1,1,0)
        self.conv_pyramid212 = nn.Conv2d(512+512+256,512,1,1,0)
        self.conv_pyramid213 = nn.Conv2d(512+512+128,512,1,1,0)
        self.conv_pyramid214 = nn.Conv2d(512+512+64,512,1,1,0)
        self.conv_pyramid223 = nn.Conv2d(512+256+128,512,1,1,0)
        self.conv_pyramid224 = nn.Conv2d(512+256+64,512,1,1,0)
        self.conv_pyramid234 = nn.Conv2d(512+128+64,512,1,1,0)
        self.conv_pyramid3234 = nn.Conv2d(512+256+128+64,512,1,1,0)
        self.conv_pyramid3134 = nn.Conv2d(512+512+128+64,512,1,1,0)
        self.conv_pyramid3124 = nn.Conv2d(512+512+256+64,512,1,1,0)
        self.conv_pyramid3123 = nn.Conv2d(512+512+256+128,512,1,1,0)
        self.conv_pyramid4 = nn.Conv2d(512+960,512,1,1,0)
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.nn1 = nn.InstanceNorm2d(512)
        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.reflecPad17_2 = nn.ReflectionPad2d((1,0,1,0))
        self.conv17 = nn.Conv2d(1024,512,3,1,0)
        if d is not None:
            self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.conv17_2 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv17_2.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17_2.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.conv19_2 = nn.Conv2d(512,256,1,1,0)
        if d is not None:
            self.conv19_2.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19_2.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.nn2 = nn.InstanceNorm2d(256)
        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.conv21_2 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv21_2.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21_2.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
            self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        if d is not None:
            self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.conv23_2 = nn.Conv2d(256,128,1,1,0)
        if d is not None:
            self.conv23_2.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23_2.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        if d is not None:
            self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
            self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)

        self.nn3 = nn.InstanceNorm2d(128)
        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(256,64,3,1,0)
        if d is not None:
            self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.conv25_2 = nn.Conv2d(128,64,3,1,0)
        if d is not None:
            self.conv25_2.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25_2.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        if d is not None:
            self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
            self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)

        self.nn4 = nn.InstanceNorm2d(64)
        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(128,3,3,1,0)
        if d is not None:
            self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())
        self.conv27_2 = nn.Conv2d(64,3,3,1,0)
        if d is not None:
            self.conv27_2.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27_2.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x, skip1, skip2, skip3, skip4, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control):
        # d0_control: transfer, skip1, skip2, skip3, skip4
        # d1_control: transfer, conv15, conv16 
        # d2_control: transfer, IN, skip+conv17, conv18, conv19, conv20, transfer
        # d3_control: transfer, IN, skip+conv21, conv22, conv23, conv24, transfer
        # d4_control: transfer, IN, skip+conv25, conv26, transfer
        # d5_control: transfer, IN, skip+conv27, transfer
        # decoder
        skip11 = self.nn01(skip1)
        skip22 = self.nn02(skip2)
        skip33 = self.nn03(skip3)
        skip44 = self.nn04(skip4)
        mid1, _ = self.maxPool_mid1(skip11)
        mid2, _ = self.maxPool_mid2(skip22)
        mid3, _ = self.maxPool_mid3(skip33)
        mid4, _ = self.maxPool_mid4(skip44)
        mid = x
        if d0_control[1] == 1:
            mid = torch.cat((mid, mid1), 1)
        if d0_control[2] == 1:
            mid = torch.cat((mid, mid2), 1)
        if d0_control[3] == 1:
            mid = torch.cat((mid, mid3), 1)
        if d0_control[4] == 1:
            mid = torch.cat((mid, mid4), 1)

        if (d0_control[1] == 0) and (d0_control[2] == 0) and (d0_control[3] == 0) and (d0_control[4] == 0):
            mid = self.conv_pyramid0(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 0) and (d0_control[3] == 0) and (d0_control[4] == 0):
            mid = self.conv_pyramid11(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 1) and (d0_control[3] == 0) and (d0_control[4] == 0):
            mid = self.conv_pyramid12(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 0) and (d0_control[3] == 1) and (d0_control[4] == 0):
            mid = self.conv_pyramid13(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 0) and (d0_control[3] == 0) and (d0_control[4] == 1):
            mid = self.conv_pyramid14(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 1) and (d0_control[3] == 0) and (d0_control[4] == 0):
            mid = self.conv_pyramid212(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 0) and (d0_control[3] == 1) and (d0_control[4] == 0):
            mid = self.conv_pyramid213(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 0) and (d0_control[3] == 0) and (d0_control[4] == 1):
            mid = self.conv_pyramid214(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 1) and (d0_control[3] == 1) and (d0_control[4] == 0):
            mid = self.conv_pyramid223(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 1) and (d0_control[3] == 0) and (d0_control[4] == 1):
            mid = self.conv_pyramid224(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 0) and (d0_control[3] == 1) and (d0_control[4] == 1):
            mid = self.conv_pyramid234(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 1) and (d0_control[3] == 1) and (d0_control[4] == 0):
            mid = self.conv_pyramid3123(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 1) and (d0_control[3] == 0) and (d0_control[4] == 1):
            mid = self.conv_pyramid3124(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 0) and (d0_control[3] == 1) and (d0_control[4] == 1):
            mid = self.conv_pyramid3134(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 1) and (d0_control[3] == 1) and (d0_control[4] == 1):
            mid = self.conv_pyramid3234(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 1) and (d0_control[3] == 1) and (d0_control[4] == 1):
            mid = self.conv_pyramid4(mid)

        if d1_control[1] == 1: 
            out = self.reflecPad15(mid)
            out = self.conv15(out)
            out = self.relu15(out)
        else:
            out = mid
        out = self.unpool(out)
        if d1_control[2] == 1:
            out = self.reflecPad16(out)
            out = self.conv16(out)
            out = self.relu16(out)

        if d2_control[1] == 1:
            skip1 = self.nn1(skip1)
        if d2_control[2] == 1:
            out = torch.cat((out, skip1), 1)
            out = self.reflecPad17(out)
            out = self.conv17(out)
            out = self.relu17(out)
        else:
            out = self.reflecPad17(out)
            out = self.conv17_2(out)
            out = self.relu17(out)
        if d2_control[3] == 1:
            out = self.reflecPad18(out)
            out = self.conv18(out)
            out = self.relu18(out)
        if d2_control[4] == 1:
            out = self.reflecPad19(out)
            out = self.conv19(out)
            out = self.relu19(out)
        else:
            # out = self.reflecPad19(out)
            out = self.conv19_2(out)
            out = self.relu19(out)
        out = self.unpool2(out)
        if d2_control[5] == 1:
            out = self.reflecPad20(out)
            out = self.conv20(out)
            out = self.relu20(out)

        if d3_control[1] == 1:
            skip2 = self.nn2(skip2)
        if d3_control[2] == 1:
            out = torch.cat((out, skip2), 1)       
            out = self.reflecPad21(out)
            out = self.conv21(out)
            out = self.relu21(out)
        else:
            out = self.reflecPad21(out)
            out = self.conv21_2(out)
            out = self.relu21(out)
        if d3_control[3] == 1:
            out = self.reflecPad22(out)
            out = self.conv22(out)
            out = self.relu22(out)
        if d3_control[4] == 1:
            out = self.reflecPad23(out)
            out = self.conv23(out)
            out = self.relu23(out)
        else:
            # out = self.reflecPad23(out)
            out = self.conv23_2(out)
            out = self.relu23(out)
        out = self.unpool3(out)
        if d3_control[5] == 1:
            out = self.reflecPad24(out)
            out = self.conv24(out)
            out = self.relu24(out)

        if d4_control[1] == 1:
            skip3 = self.nn3(skip3)
        if d4_control[2] == 1:
            out = torch.cat((out, skip3), 1)
            out = self.reflecPad25(out)
            out = self.conv25(out)
            out = self.relu25(out)
        else:
            out = self.reflecPad25(out)
            out = self.conv25_2(out)
            out = self.relu25(out)
        out = self.unpool4(out)
        if d4_control[3] == 1:
            out = self.reflecPad26(out)
            out = self.conv26(out)
            out = self.relu26(out)

        
        if d5_control[1] == 1:
            skip4 = self.nn4(skip4)
        if d5_control[2] == 1:
            out = torch.cat((out, skip4), 1)
            out = self.reflecPad27(out)
            out = self.conv27(out)
        else:
            out = self.reflecPad27(out)
            out = self.conv27_2(out)

        return out

class decoder0(nn.Module):
    def __init__(self,d=None):
        super(decoder0,self).__init__()
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=4,stride=4,return_indices = True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=8,stride=8,return_indices = True)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=16,stride=16,return_indices = True)
        self.nn01 = nn.InstanceNorm2d(512)
        self.nn02 = nn.InstanceNorm2d(256)
        self.nn03 = nn.InstanceNorm2d(128)
        self.nn04 = nn.InstanceNorm2d(64)

        self.conv_pyramid0 = nn.Conv2d(512+0,512,1,1,0)
        self.conv_pyramid11 = nn.Conv2d(512+512,512,1,1,0)
        self.conv_pyramid12 = nn.Conv2d(512+256,512,1,1,0)
        self.conv_pyramid13 = nn.Conv2d(512+128,512,1,1,0)
        self.conv_pyramid14 = nn.Conv2d(512+64,512,1,1,0)
        self.conv_pyramid212 = nn.Conv2d(512+512+256,512,1,1,0)
        self.conv_pyramid213 = nn.Conv2d(512+512+128,512,1,1,0)
        self.conv_pyramid214 = nn.Conv2d(512+512+64,512,1,1,0)
        self.conv_pyramid223 = nn.Conv2d(512+256+128,512,1,1,0)
        self.conv_pyramid224 = nn.Conv2d(512+256+64,512,1,1,0)
        self.conv_pyramid234 = nn.Conv2d(512+128+64,512,1,1,0)
        self.conv_pyramid3234 = nn.Conv2d(512+256+128+64,512,1,1,0)
        self.conv_pyramid3134 = nn.Conv2d(512+512+128+64,512,1,1,0)
        self.conv_pyramid3124 = nn.Conv2d(512+512+256+64,512,1,1,0)
        self.conv_pyramid3123 = nn.Conv2d(512+512+256+128,512,1,1,0)
        self.conv_pyramid4 = nn.Conv2d(512+960,512,1,1,0)
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.nn1 = nn.InstanceNorm2d(512)
        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(1024,512,3,1,0)
        if d is not None:
            self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.conv17_2 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv17_2.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17_2.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.conv19_2 = nn.Conv2d(512,256,1,1,0)
        if d is not None:
            self.conv19_2.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19_2.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.nn2 = nn.InstanceNorm2d(256)
        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.conv21_2 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv21_2.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21_2.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
            self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        if d is not None:
            self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.conv23_2 = nn.Conv2d(256,128,1,1,0)
        if d is not None:
            self.conv23_2.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23_2.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        if d is not None:
            self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
            self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)

        self.nn3 = nn.InstanceNorm2d(128)
        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(256,64,3,1,0)
        if d is not None:
            self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.conv25_2 = nn.Conv2d(128,64,3,1,0)
        if d is not None:
            self.conv25_2.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25_2.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        if d is not None:
            self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
            self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)

        self.nn4 = nn.InstanceNorm2d(64)
        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(128,3,3,1,0)
        if d is not None:
            self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())
        self.conv27_2 = nn.Conv2d(64,3,3,1,0)
        if d is not None:
            self.conv27_2.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27_2.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x, skip1, skip2, skip3, skip4, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control):
        # decoder
        skip11 = self.nn01(skip1)
        skip22 = self.nn02(skip2)
        skip33 = self.nn03(skip3)
        skip44 = self.nn04(skip4)
        mid1, _ = self.maxPool_mid1(skip11)
        mid2, _ = self.maxPool_mid2(skip22)
        mid3, _ = self.maxPool_mid3(skip33)
        mid4, _ = self.maxPool_mid4(skip44)
        mid = x
        if d0_control[1] == 1:
            mid = torch.cat((mid, mid1), 1)
        if d0_control[2] == 1:
            mid = torch.cat((mid, mid2), 1)
        if d0_control[3] == 1:
            mid = torch.cat((mid, mid3), 1)
        if d0_control[4] == 1:
            mid = torch.cat((mid, mid4), 1)

        if (d0_control[1] == 0) and (d0_control[2] == 0) and (d0_control[3] == 0) and (d0_control[4] == 0):
            mid = self.conv_pyramid0(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 0) and (d0_control[3] == 0) and (d0_control[4] == 0):
            mid = self.conv_pyramid11(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 1) and (d0_control[3] == 0) and (d0_control[4] == 0):
            mid = self.conv_pyramid12(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 0) and (d0_control[3] == 1) and (d0_control[4] == 0):
            mid = self.conv_pyramid13(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 0) and (d0_control[3] == 0) and (d0_control[4] == 1):
            mid = self.conv_pyramid14(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 1) and (d0_control[3] == 0) and (d0_control[4] == 0):
            mid = self.conv_pyramid212(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 0) and (d0_control[3] == 1) and (d0_control[4] == 0):
            mid = self.conv_pyramid213(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 0) and (d0_control[3] == 0) and (d0_control[4] == 1):
            mid = self.conv_pyramid214(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 1) and (d0_control[3] == 1) and (d0_control[4] == 0):
            mid = self.conv_pyramid223(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 1) and (d0_control[3] == 0) and (d0_control[4] == 1):
            mid = self.conv_pyramid224(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 0) and (d0_control[3] == 1) and (d0_control[4] == 1):
            mid = self.conv_pyramid234(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 1) and (d0_control[3] == 1) and (d0_control[4] == 0):
            mid = self.conv_pyramid3123(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 1) and (d0_control[3] == 0) and (d0_control[4] == 1):
            mid = self.conv_pyramid3124(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 0) and (d0_control[3] == 1) and (d0_control[4] == 1):
            mid = self.conv_pyramid3134(mid)
        if (d0_control[1] == 0) and (d0_control[2] == 1) and (d0_control[3] == 1) and (d0_control[4] == 1):
            mid = self.conv_pyramid3234(mid)
        if (d0_control[1] == 1) and (d0_control[2] == 1) and (d0_control[3] == 1) and (d0_control[4] == 1):
            mid = self.conv_pyramid4(mid)
        return mid

class decoder1(nn.Module):
    def __init__(self,d=None):
        super(decoder1,self).__init__()
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=4,stride=4,return_indices = True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=8,stride=8,return_indices = True)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=16,stride=16,return_indices = True)
        self.nn01 = nn.InstanceNorm2d(512)
        self.nn02 = nn.InstanceNorm2d(256)
        self.nn03 = nn.InstanceNorm2d(128)
        self.nn04 = nn.InstanceNorm2d(64)

        self.conv_pyramid0 = nn.Conv2d(512+0,512,1,1,0)
        self.conv_pyramid11 = nn.Conv2d(512+512,512,1,1,0)
        self.conv_pyramid12 = nn.Conv2d(512+256,512,1,1,0)
        self.conv_pyramid13 = nn.Conv2d(512+128,512,1,1,0)
        self.conv_pyramid14 = nn.Conv2d(512+64,512,1,1,0)
        self.conv_pyramid212 = nn.Conv2d(512+512+256,512,1,1,0)
        self.conv_pyramid213 = nn.Conv2d(512+512+128,512,1,1,0)
        self.conv_pyramid214 = nn.Conv2d(512+512+64,512,1,1,0)
        self.conv_pyramid223 = nn.Conv2d(512+256+128,512,1,1,0)
        self.conv_pyramid224 = nn.Conv2d(512+256+64,512,1,1,0)
        self.conv_pyramid234 = nn.Conv2d(512+128+64,512,1,1,0)
        self.conv_pyramid3234 = nn.Conv2d(512+256+128+64,512,1,1,0)
        self.conv_pyramid3134 = nn.Conv2d(512+512+128+64,512,1,1,0)
        self.conv_pyramid3124 = nn.Conv2d(512+512+256+64,512,1,1,0)
        self.conv_pyramid3123 = nn.Conv2d(512+512+256+128,512,1,1,0)
        self.conv_pyramid4 = nn.Conv2d(512+960,512,1,1,0)
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.nn1 = nn.InstanceNorm2d(512)
        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(1024,512,3,1,0)
        if d is not None:
            self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.conv17_2 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv17_2.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17_2.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.conv19_2 = nn.Conv2d(512,256,1,1,0)
        if d is not None:
            self.conv19_2.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19_2.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.nn2 = nn.InstanceNorm2d(256)
        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.conv21_2 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv21_2.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21_2.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
            self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        if d is not None:
            self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.conv23_2 = nn.Conv2d(256,128,1,1,0)
        if d is not None:
            self.conv23_2.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23_2.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        if d is not None:
            self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
            self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)

        self.nn3 = nn.InstanceNorm2d(128)
        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(256,64,3,1,0)
        if d is not None:
            self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.conv25_2 = nn.Conv2d(128,64,3,1,0)
        if d is not None:
            self.conv25_2.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25_2.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        if d is not None:
            self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
            self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)

        self.nn4 = nn.InstanceNorm2d(64)
        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(128,3,3,1,0)
        if d is not None:
            self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())
        self.conv27_2 = nn.Conv2d(64,3,3,1,0)
        if d is not None:
            self.conv27_2.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27_2.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x, skip1, skip2, skip3, skip4, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control):
        mid = x
        if d1_control[1] == 1: 
            out = self.reflecPad15(mid)
            out = self.conv15(out)
            out = self.relu15(out)
        else:
            out = mid
        out = self.unpool(out)
        if d1_control[2] == 1:
            out = self.reflecPad16(out)
            out = self.conv16(out)
            out = self.relu16(out)
        return out

class decoder2(nn.Module):
    def __init__(self,d=None):
        super(decoder2,self).__init__()
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=4,stride=4,return_indices = True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=8,stride=8,return_indices = True)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=16,stride=16,return_indices = True)
        self.nn01 = nn.InstanceNorm2d(512)
        self.nn02 = nn.InstanceNorm2d(256)
        self.nn03 = nn.InstanceNorm2d(128)
        self.nn04 = nn.InstanceNorm2d(64)

        self.conv_pyramid0 = nn.Conv2d(512+0,512,1,1,0)
        self.conv_pyramid11 = nn.Conv2d(512+512,512,1,1,0)
        self.conv_pyramid12 = nn.Conv2d(512+256,512,1,1,0)
        self.conv_pyramid13 = nn.Conv2d(512+128,512,1,1,0)
        self.conv_pyramid14 = nn.Conv2d(512+64,512,1,1,0)
        self.conv_pyramid212 = nn.Conv2d(512+512+256,512,1,1,0)
        self.conv_pyramid213 = nn.Conv2d(512+512+128,512,1,1,0)
        self.conv_pyramid214 = nn.Conv2d(512+512+64,512,1,1,0)
        self.conv_pyramid223 = nn.Conv2d(512+256+128,512,1,1,0)
        self.conv_pyramid224 = nn.Conv2d(512+256+64,512,1,1,0)
        self.conv_pyramid234 = nn.Conv2d(512+128+64,512,1,1,0)
        self.conv_pyramid3234 = nn.Conv2d(512+256+128+64,512,1,1,0)
        self.conv_pyramid3134 = nn.Conv2d(512+512+128+64,512,1,1,0)
        self.conv_pyramid3124 = nn.Conv2d(512+512+256+64,512,1,1,0)
        self.conv_pyramid3123 = nn.Conv2d(512+512+256+128,512,1,1,0)
        self.conv_pyramid4 = nn.Conv2d(512+960,512,1,1,0)
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.nn1 = nn.InstanceNorm2d(512)
        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(1024,512,3,1,0)
        if d is not None:
            self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.conv17_2 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv17_2.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17_2.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.conv19_2 = nn.Conv2d(512,256,1,1,0)
        if d is not None:
            self.conv19_2.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19_2.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.nn2 = nn.InstanceNorm2d(256)
        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.conv21_2 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv21_2.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21_2.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
            self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        if d is not None:
            self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.conv23_2 = nn.Conv2d(256,128,1,1,0)
        if d is not None:
            self.conv23_2.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23_2.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        if d is not None:
            self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
            self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)

        self.nn3 = nn.InstanceNorm2d(128)
        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(256,64,3,1,0)
        if d is not None:
            self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.conv25_2 = nn.Conv2d(128,64,3,1,0)
        if d is not None:
            self.conv25_2.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25_2.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        if d is not None:
            self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
            self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)

        self.nn4 = nn.InstanceNorm2d(64)
        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(128,3,3,1,0)
        if d is not None:
            self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())
        self.conv27_2 = nn.Conv2d(64,3,3,1,0)
        if d is not None:
            self.conv27_2.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27_2.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x, skip1, skip2, skip3, skip4, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control):
        # decoder
        out = x
        if d2_control[1] == 1:
            skip1 = self.nn1(skip1)
        if d2_control[2] == 1:
            out = torch.cat((out, skip1), 1)
            out = self.reflecPad17(out)
            out = self.conv17(out)
            out = self.relu17(out)
        else:
            out = self.reflecPad17(out)
            out = self.conv17_2(out)
            out = self.relu17(out)
        if d2_control[3] == 1:
            out = self.reflecPad18(out)
            out = self.conv18(out)
            out = self.relu18(out)
        if d2_control[4] == 1:
            out = self.reflecPad19(out)
            out = self.conv19(out)
            out = self.relu19(out)
        else:
            # out = self.reflecPad19(out)
            out = self.conv19_2(out)
            out = self.relu19(out)
        out = self.unpool2(out)
        if d2_control[5] == 1:
            out = self.reflecPad20(out)
            out = self.conv20(out)
            out = self.relu20(out)
        return out

class decoder3(nn.Module):
    def __init__(self,d=None):
        super(decoder3,self).__init__()
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=4,stride=4,return_indices = True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=8,stride=8,return_indices = True)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=16,stride=16,return_indices = True)
        self.nn01 = nn.InstanceNorm2d(512)
        self.nn02 = nn.InstanceNorm2d(256)
        self.nn03 = nn.InstanceNorm2d(128)
        self.nn04 = nn.InstanceNorm2d(64)

        self.conv_pyramid0 = nn.Conv2d(512+0,512,1,1,0)
        self.conv_pyramid11 = nn.Conv2d(512+512,512,1,1,0)
        self.conv_pyramid12 = nn.Conv2d(512+256,512,1,1,0)
        self.conv_pyramid13 = nn.Conv2d(512+128,512,1,1,0)
        self.conv_pyramid14 = nn.Conv2d(512+64,512,1,1,0)
        self.conv_pyramid212 = nn.Conv2d(512+512+256,512,1,1,0)
        self.conv_pyramid213 = nn.Conv2d(512+512+128,512,1,1,0)
        self.conv_pyramid214 = nn.Conv2d(512+512+64,512,1,1,0)
        self.conv_pyramid223 = nn.Conv2d(512+256+128,512,1,1,0)
        self.conv_pyramid224 = nn.Conv2d(512+256+64,512,1,1,0)
        self.conv_pyramid234 = nn.Conv2d(512+128+64,512,1,1,0)
        self.conv_pyramid3234 = nn.Conv2d(512+256+128+64,512,1,1,0)
        self.conv_pyramid3134 = nn.Conv2d(512+512+128+64,512,1,1,0)
        self.conv_pyramid3124 = nn.Conv2d(512+512+256+64,512,1,1,0)
        self.conv_pyramid3123 = nn.Conv2d(512+512+256+128,512,1,1,0)
        self.conv_pyramid4 = nn.Conv2d(512+960,512,1,1,0)
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.nn1 = nn.InstanceNorm2d(512)
        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(1024,512,3,1,0)
        if d is not None:
            self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.conv17_2 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv17_2.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17_2.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.conv19_2 = nn.Conv2d(512,256,1,1,0)
        if d is not None:
            self.conv19_2.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19_2.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.nn2 = nn.InstanceNorm2d(256)
        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.conv21_2 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv21_2.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21_2.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
            self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        if d is not None:
            self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.conv23_2 = nn.Conv2d(256,128,1,1,0)
        if d is not None:
            self.conv23_2.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23_2.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        if d is not None:
            self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
            self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)

        self.nn3 = nn.InstanceNorm2d(128)
        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(256,64,3,1,0)
        if d is not None:
            self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.conv25_2 = nn.Conv2d(128,64,3,1,0)
        if d is not None:
            self.conv25_2.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25_2.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        if d is not None:
            self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
            self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)

        self.nn4 = nn.InstanceNorm2d(64)
        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(128,3,3,1,0)
        if d is not None:
            self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())
        self.conv27_2 = nn.Conv2d(64,3,3,1,0)
        if d is not None:
            self.conv27_2.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27_2.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x, skip1, skip2, skip3, skip4, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control):
       # decoder
        out = x
        if d3_control[1] == 1:
            skip2 = self.nn2(skip2)
        if d3_control[2] == 1:
            out = torch.cat((out, skip2), 1)       
            out = self.reflecPad21(out)
            out = self.conv21(out)
            out = self.relu21(out)
        else:
            out = self.reflecPad21(out)
            out = self.conv21_2(out)
            out = self.relu21(out)
        if d3_control[3] == 1:
            out = self.reflecPad22(out)
            out = self.conv22(out)
            out = self.relu22(out)
        if d3_control[4] == 1:
            out = self.reflecPad23(out)
            out = self.conv23(out)
            out = self.relu23(out)
        else:
            # out = self.reflecPad23(out)
            out = self.conv23_2(out)
            out = self.relu23(out)
        out = self.unpool3(out)
        if d3_control[5] == 1:
            out = self.reflecPad24(out)
            out = self.conv24(out)
            out = self.relu24(out)
        return out

class decoder4(nn.Module):
    def __init__(self,d=None):
        super(decoder4,self).__init__()
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=4,stride=4,return_indices = True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=8,stride=8,return_indices = True)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=16,stride=16,return_indices = True)
        self.nn01 = nn.InstanceNorm2d(512)
        self.nn02 = nn.InstanceNorm2d(256)
        self.nn03 = nn.InstanceNorm2d(128)
        self.nn04 = nn.InstanceNorm2d(64)

        self.conv_pyramid0 = nn.Conv2d(512+0,512,1,1,0)
        self.conv_pyramid11 = nn.Conv2d(512+512,512,1,1,0)
        self.conv_pyramid12 = nn.Conv2d(512+256,512,1,1,0)
        self.conv_pyramid13 = nn.Conv2d(512+128,512,1,1,0)
        self.conv_pyramid14 = nn.Conv2d(512+64,512,1,1,0)
        self.conv_pyramid212 = nn.Conv2d(512+512+256,512,1,1,0)
        self.conv_pyramid213 = nn.Conv2d(512+512+128,512,1,1,0)
        self.conv_pyramid214 = nn.Conv2d(512+512+64,512,1,1,0)
        self.conv_pyramid223 = nn.Conv2d(512+256+128,512,1,1,0)
        self.conv_pyramid224 = nn.Conv2d(512+256+64,512,1,1,0)
        self.conv_pyramid234 = nn.Conv2d(512+128+64,512,1,1,0)
        self.conv_pyramid3234 = nn.Conv2d(512+256+128+64,512,1,1,0)
        self.conv_pyramid3134 = nn.Conv2d(512+512+128+64,512,1,1,0)
        self.conv_pyramid3124 = nn.Conv2d(512+512+256+64,512,1,1,0)
        self.conv_pyramid3123 = nn.Conv2d(512+512+256+128,512,1,1,0)
        self.conv_pyramid4 = nn.Conv2d(512+960,512,1,1,0)
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.nn1 = nn.InstanceNorm2d(512)
        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(1024,512,3,1,0)
        if d is not None:
            self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.conv17_2 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv17_2.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17_2.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.conv19_2 = nn.Conv2d(512,256,1,1,0)
        if d is not None:
            self.conv19_2.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19_2.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.nn2 = nn.InstanceNorm2d(256)
        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.conv21_2 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv21_2.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21_2.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
            self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        if d is not None:
            self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.conv23_2 = nn.Conv2d(256,128,1,1,0)
        if d is not None:
            self.conv23_2.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23_2.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        if d is not None:
            self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
            self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)

        self.nn3 = nn.InstanceNorm2d(128)
        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(256,64,3,1,0)
        if d is not None:
            self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.conv25_2 = nn.Conv2d(128,64,3,1,0)
        if d is not None:
            self.conv25_2.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25_2.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        if d is not None:
            self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
            self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)

        self.nn4 = nn.InstanceNorm2d(64)
        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(128,3,3,1,0)
        if d is not None:
            self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())
        self.conv27_2 = nn.Conv2d(64,3,3,1,0)
        if d is not None:
            self.conv27_2.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27_2.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x, skip1, skip2, skip3, skip4, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control):
        # decoder
        out = x
        if d4_control[1] == 1:
            skip3 = self.nn3(skip3)
        if d4_control[2] == 1:
            out = torch.cat((out, skip3), 1)
            out = self.reflecPad25(out)
            out = self.conv25(out)
            out = self.relu25(out)
        else:
            out = self.reflecPad25(out)
            out = self.conv25_2(out)
            out = self.relu25(out)
        out = self.unpool4(out)
        if d4_control[3] == 1:
            out = self.reflecPad26(out)
            out = self.conv26(out)
            out = self.relu26(out)
        return out

class decoder5(nn.Module):
    def __init__(self,d=None):
        super(decoder5,self).__init__()
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=4,stride=4,return_indices = True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=8,stride=8,return_indices = True)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=16,stride=16,return_indices = True)
        self.nn01 = nn.InstanceNorm2d(512)
        self.nn02 = nn.InstanceNorm2d(256)
        self.nn03 = nn.InstanceNorm2d(128)
        self.nn04 = nn.InstanceNorm2d(64)

        self.conv_pyramid0 = nn.Conv2d(512+0,512,1,1,0)
        self.conv_pyramid11 = nn.Conv2d(512+512,512,1,1,0)
        self.conv_pyramid12 = nn.Conv2d(512+256,512,1,1,0)
        self.conv_pyramid13 = nn.Conv2d(512+128,512,1,1,0)
        self.conv_pyramid14 = nn.Conv2d(512+64,512,1,1,0)
        self.conv_pyramid212 = nn.Conv2d(512+512+256,512,1,1,0)
        self.conv_pyramid213 = nn.Conv2d(512+512+128,512,1,1,0)
        self.conv_pyramid214 = nn.Conv2d(512+512+64,512,1,1,0)
        self.conv_pyramid223 = nn.Conv2d(512+256+128,512,1,1,0)
        self.conv_pyramid224 = nn.Conv2d(512+256+64,512,1,1,0)
        self.conv_pyramid234 = nn.Conv2d(512+128+64,512,1,1,0)
        self.conv_pyramid3234 = nn.Conv2d(512+256+128+64,512,1,1,0)
        self.conv_pyramid3134 = nn.Conv2d(512+512+128+64,512,1,1,0)
        self.conv_pyramid3124 = nn.Conv2d(512+512+256+64,512,1,1,0)
        self.conv_pyramid3123 = nn.Conv2d(512+512+256+128,512,1,1,0)
        self.conv_pyramid4 = nn.Conv2d(512+960,512,1,1,0)
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.nn1 = nn.InstanceNorm2d(512)
        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(1024,512,3,1,0)
        if d is not None:
            self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.conv17_2 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv17_2.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17_2.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        if d is not None:
            self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.conv19_2 = nn.Conv2d(512,256,1,1,0)
        if d is not None:
            self.conv19_2.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19_2.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.nn2 = nn.InstanceNorm2d(256)
        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(512,256,3,1,0)
        if d is not None:
            self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.conv21_2 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv21_2.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21_2.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        if d is not None:
            self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
            self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        if d is not None:
            self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.conv23_2 = nn.Conv2d(256,128,1,1,0)
        if d is not None:
            self.conv23_2.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23_2.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        if d is not None:
            self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
            self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)

        self.nn3 = nn.InstanceNorm2d(128)
        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(256,64,3,1,0)
        if d is not None:
            self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.conv25_2 = nn.Conv2d(128,64,3,1,0)
        if d is not None:
            self.conv25_2.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25_2.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        if d is not None:
            self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
            self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)

        self.nn4 = nn.InstanceNorm2d(64)
        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(128,3,3,1,0)
        if d is not None:
            self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())
        self.conv27_2 = nn.Conv2d(64,3,3,1,0)
        if d is not None:
            self.conv27_2.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27_2.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x, skip1, skip2, skip3, skip4, d0_control, d1_control, d2_control, d3_control, d4_control, d5_control):
        # decoder
        out = x
        if d5_control[1] == 1:
            skip4 = self.nn4(skip4)
        if d5_control[2] == 1:
            out = torch.cat((out, skip4), 1)
            out = self.reflecPad27(out)
            out = self.conv27(out)
        else:
            out = self.reflecPad27(out)
            out = self.conv27_2(out)
        return out

if __name__ == '__main__':
    embed()
