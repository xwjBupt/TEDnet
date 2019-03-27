
"""
create by xwj
"""

import torchvision as vision
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.SSIM import SSIM
from collections import OrderedDict as Order


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False,use_drop = False, **kwargs):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.use_drop = use_drop
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None
        self.drop = nn.Dropout(p = 0.5,inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_drop:
            x = self.drop(x)

        return F.relu(x, inplace=True)

class MSA_block(nn.Module):
    def __init__(self,in_channels,out_channels,use_bn = True,level = 1,use_drop = True ,**kwargs):
        super(MSA_block,self).__init__()
        branch_out = in_channels//2
        dia = level
        self.use_drop = use_drop
        pad3 = (2*(dia-1)+2)//2
        pad5 = (4*(dia-1)+4)//2
        pad7 = (6*(dia-1)+6)//2
        self.branch1x1 = BasicConv(in_channels, out_channels, self.use_drop,use_bn=use_bn,kernel_size=1)

        self.branch3x3 = nn.Sequential(
            BasicConv(in_channels,branch_out, self.use_drop,use_bn=use_bn,
                      kernel_size=1),
            BasicConv(branch_out, out_channels,self.use_drop, use_bn=use_bn,
                      kernel_size=3, padding=pad3,dilation = dia),
        )
        self.branch5x5 = nn.Sequential(
            BasicConv(in_channels, branch_out,self.use_drop, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(branch_out, out_channels,self.use_drop, use_bn=use_bn,
                      kernel_size=5, padding=pad5,dilation = dia),
        )
        self.branch7x7 = nn.Sequential(
            BasicConv(in_channels, branch_out, self.use_drop,use_bn=use_bn,
                      kernel_size=1),
            BasicConv(branch_out, out_channels,self.use_drop, use_bn=use_bn,
                      kernel_size=7, padding=pad7,dilation = dia),
        )

        self.squeese = nn.Sequential(
            BasicConv(3*out_channels,out_channels,self.use_drop,use_bn=use_bn,kernel_size = 1,padding = 0)
        )



    def forward(self, x):

            branch3x3 = self.branch3x3(x)
            branch5x5 = self.branch5x5(x)
            branch7x7 = self.branch7x7(x)
            out = torch.cat([branch3x3,branch5x5,branch7x7],1)
            fuse = self.squeese(out)
            result = torch.cat([fuse,x],1)

            return result



class DecodingBlock(nn.Module):
    def __init__(self,low,high,out,**kwargs):
        super(DecodingBlock,self).__init__()
        lowin = low
        highin = high
        self.F1 = BasicConv(lowin,highin,use_bn=True,kernel_size = 1,use_drop=True)
        self.F2 = nn.ConvTranspose2d(highin,highin,kernel_size=3,padding=1)
        self.F3 = BasicConv(2* highin ,out,use_bn=True,kernel_size = 1,use_drop=True)
        self.upsample = UpsamleBlock(highin,highin)

    def forward(self,low_feature,high_feature):
        f1_out = self.F1(low_feature)
        up = self.upsample(high_feature)
        f3_in = torch.cat([f1_out,up],1)
        f3_out = self.F3(f3_in)

        return f3_out

class UpsamleBlock(nn.Module):
    def __init__(self,in_channels,out_channels ):
        super(UpsamleBlock, self).__init__()
        self.conv = BasicConv(in_channels,out_channels,kernel_size = 3,stride = 1,padding = 1,use_drop=True)

    def forward(self, x):
        x = F.interpolate(x,[x.shape[2]*2,x.shape[3]*2],mode='nearest')
        x = self.conv(x)
        return x



class Simple_Triangle(nn.Module):
    def __init__(self,gray = False,use_bn = True,use_drop = True,log = None,in_channels = 3):
        super(Simple_Triangle,self).__init__()
        if gray:
            in_channels = 1
        else:
            in_channels = 3
        if log is not None:
            self.log = log
        self.use_drop = use_drop
        self.base = nn.Sequential(Order([
            ('conv1',BasicConv(in_channels,out_channels=32,use_bn=True,use_drop=True,kernel_size = 3,padding = 1)),
            ('conv2', BasicConv(in_channels=32, out_channels=64, use_bn=True, kernel_size=3,padding = 1 ))
        ]))

        self.MSA1 = nn.Sequential(
            MSA_block(in_channels=64, out_channels=64,use_bn=True,level=1),
            nn.MaxPool2d(2, 2)
        )

        self.MSA2 = nn.Sequential(
            MSA_block(in_channels=128, out_channels=128,use_bn=True,level=2),
            nn.MaxPool2d(2, 2)
        )

        self.MSA3 = nn.Sequential(
            MSA_block(in_channels=256, out_channels=256,use_bn=True,level=3),
            nn.MaxPool2d(2, 2)
        )

        self.Decoding1_1 = DecodingBlock(low = 256,high = 512,out = 256)
        self.Decoding1_2 = DecodingBlock(low=128, high=256, out=128)
        self.Decoding2_1 = DecodingBlock(low = 128,high = 256,out = 64)

        self.upsample1_1 = UpsamleBlock(64,8)
        self.FinalConv1 = BasicConv(in_channels=8,out_channels=3,use_bn=False,kernel_size=3,stride=1,padding=1)


    def  forward(self, x):
        x0 = self.base(x)
        x1 = self.MSA1(x0)
        x2 = self.MSA2(x1)
        x3 = self.MSA3(x2)
        decode1_1 = self.Decoding1_1(x2,x3)
        decode1_2 = self.Decoding1_2(x1,x2)
        decode2_1 = self.Decoding2_1(decode1_2,decode1_1)

        up = self.upsample1_1(decode2_1)
        es_map =self.FinalConv1(up)

        return es_map



class Triangle(nn.Module):
    def __init__(self,gray = False,log = None,use_bn = True,training = True):
        super(Triangle,self).__init__()
        if gray:
            in_channels = 1
        else:
            in_channels = 3
        if log is not None:
            self.log = log

        self.base = nn.Sequential(Order([
            ('conv1',BasicConv(in_channels = 3,out_channels=32,use_bn=True,kernel_size = 3,padding = 1)),
            ('conv2', BasicConv(in_channels=32, out_channels=64, use_bn=True, kernel_size=3,padding = 1))
        ]))

        self.MSA1 = nn.Sequential(
            MSA_block(in_channels=64, out_channels=64,use_bn=True,level=1),
            nn.MaxPool2d(2, 2)
        )

        self.MSA2 = nn.Sequential(
            MSA_block(in_channels=128, out_channels=128,use_bn=True,level=2),
            nn.MaxPool2d(2, 2)
        )

        self.MSA3 = nn.Sequential(
            MSA_block(in_channels=256, out_channels=256,use_bn=True,level=3),
            nn.MaxPool2d(2, 2)
        )

        self.Decoding1_1 = DecodingBlock(low = 256,high = 512,out = 256)
        self.Decoding1_2 = DecodingBlock(low=128, high=256, out=128)
        self.Decoding2_1 = DecodingBlock(low = 128,high = 256,out = 64)

        self.upsample1_1 = UpsamleBlock(64,8)
        self.FinalConv1 = BasicConv(in_channels=8,out_channels=3,use_bn=False,kernel_size=3,stride=1,padding=1)


    def  forward(self, x):
        x0 = self.base(x)
        x1 = self.MSA1(x0)
        x2 = self.MSA2(x1)
        x3 = self.MSA3(x2)
        decode1_1 = self.Decoding1_1(x2,x3)
        decode1_2 = self.Decoding1_2(x1,x2)
        decode2_1 = self.Decoding2_1(decode1_2,decode1_1)

        up = self.upsample1_1(decode2_1)
        es_map =self.FinalConv1(up)

        return es_map







class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.LE = nn.MSELoss(reduction='elementwise_mean')
        self.SSIM = SSIM(in_channel=3, window_size=11, size_average=True)
        self.loss_C = 0.
        self.loss_E = 0.

    def forward(self,es_map,gt_map):
        self.loss_E = self.LE(es_map, gt_map)
        self.loss_C = 1 - self.SSIM(es_map, gt_map)
        my_loss = self.loss_E + 0.001 * self.loss_C

        return my_loss




