import torch
import torch.nn as nn
import math
from model import common


def make_model(args, parent=False):
    return ADC16_12(args)

def abandon():
    return ['skip','tail']

class Scale(nn.Module):

    def __init__(self,init_value=1e-3,requires_grad=True):
        super().__init__()
        self.cpu = False
        self.scale = nn.Parameter(torch.FloatTensor([init_value]),requires_grad=requires_grad)

    def forward(self, input):
        if self.cpu:
            return input * self.scale
        else :
           return input * self.scale.cuda()

class WDU(nn.Module): #CONV UNIT
    def __init__(
        self, n_feats, kernel_size, block_feats, wn,  act=nn.ReLU(True),weight_init=1):
        super(WDU, self).__init__()
        body = []
        body.append(
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        wdu = self.body(x)

        return wdu


class WDB_B(nn.Module): #ADRB
    def __init__(
        self, n_feats, kernel_size, block_feats, wn, alpha=1.0,beta=1.0, act=nn.ReLU(True), weight_init=1):
        super(WDB_B, self).__init__()
        a01,a12,a23,a34 = alpha,alpha,alpha,alpha
        b01,b02,b03,b04 = beta,beta,beta,beta
        b12,b13,b14     = beta,beta,beta
        b23,b24         = beta,beta
        b34             = beta

        self.A01 = Scale(a01)
        self.A12 = Scale(a12)
        self.A23 = Scale(a23)
        self.A34 = Scale(a34)

        self.B01 = Scale(b01)
        self.B02 = Scale(b02)
        self.B03 = Scale(b03)
        self.B04 = Scale(b04)

        self.B12 = Scale(b12)
        self.B13 = Scale(b13)
        self.B14 = Scale(b14)

        self.B23 = Scale(b23)
        self.B24 = Scale(b24)

        self.B34 = Scale(b34)

        self.get_Y1=WDU(n_feats,kernel_size,block_feats,wn,act=act,weight_init=weight_init)
        self.get_Y2=WDU(n_feats,kernel_size,block_feats,wn,act=act,weight_init=weight_init)
        self.get_Y3=WDU(n_feats,kernel_size,block_feats,wn,act=act,weight_init=weight_init)
        self.get_Y4=WDU(n_feats,kernel_size,block_feats,wn,act=act,weight_init=weight_init)
        self.LFF_B = nn.Conv2d(5*n_feats, n_feats, 1, padding=0, stride=1)

    def forward(self, x):
        Y1=self.get_Y1(x)
        X1=self.A01(Y1)+self.B01(x)

        Y2=self.get_Y2(X1)
        X2=self.A12(Y2)+self.B12(Y1)+self.B02(x)

        Y3=self.get_Y3(X2)
        X3=self.A23(Y3)+self.B23(Y2)+self.B13(Y1)+self.B03(x)

        Y4=self.get_Y4(X3)

        X4=[]
        X4.append(self.A34(Y4))
        X4.append(self.B34(Y3))
        X4.append(self.B24(Y2))
        X4.append(self.B14(Y1))
        X4.append(self.B04(x))
        Y5=torch.cat(X4,1)

        return self.LFF_B(Y5)+x

  

class WDN_B(nn.Module): #ADRU
    def __init__(self, n_feats, kernel_size, block_feats, wn, alpha=1.0,beta=1.0, act=nn.ReLU(True),weight_init=1):
        super(WDN_B, self).__init__()

        a01,a12,a23,a34 = alpha,alpha,alpha,alpha
        b01,b02,b03,b04 = beta,beta,beta,beta
        b12,b13,b14     = beta,beta,beta
        b23,b24         = beta,beta
        b34             = beta

        self.A01 = Scale(a01)
        self.A12 = Scale(a12)
        self.A23 = Scale(a23)
        self.A34 = Scale(a34)

        self.B01 = Scale(b01)
        self.B02 = Scale(b02)
        self.B03 = Scale(b03)
        self.B04 = Scale(b04)

        self.B12 = Scale(b12)
        self.B13 = Scale(b13)
        self.B14 = Scale(b14)

        self.B23 = Scale(b23)
        self.B24 = Scale(b24)

        self.B34 = Scale(b34)


        self.wdb1=WDB_B(n_feats, kernel_size, block_feats, wn, alpha, beta,act=act,weight_init=weight_init)
        self.wdb2=WDB_B(n_feats, kernel_size, block_feats, wn, alpha, beta,act=act,weight_init=weight_init)
        self.wdb3=WDB_B(n_feats, kernel_size, block_feats, wn, alpha, beta,act=act,weight_init=weight_init)
        self.wdb4=WDB_B(n_feats, kernel_size, block_feats, wn, alpha, beta,act=act,weight_init=weight_init)
        self.LFF = nn.Conv2d(5*n_feats, n_feats, 1, padding=0, stride=1)

    def forward(self, x):

        Y1=self.wdb1(x)
        X1=self.A01(Y1)+self.B01(x)

        Y2=self.wdb2(X1)
        X2=self.A12(Y2)+self.B12(Y1)+self.B02(x)

        Y3=self.wdb3(X2)
        X3=self.A23(Y3)+self.B23(Y2)+self.B13(Y1)+self.B03(x)

        Y4=self.wdb4(X3)
        X4=[]
        X4.append(self.A34(Y4))
        X4.append(self.B34(Y3))
        X4.append(self.B24(Y2))
        X4.append(self.B14(Y1))
        X4.append(self.B04(x))
        Y5=torch.cat(X4,1)

        return self.LFF(Y5)+x

    

class FFSC(nn.Module): #AFSL
    def __init__(
        self, args, scale, n_feats, kernel_size, wn, weight_init=1):
        super(FFSC, self).__init__()
        self.scale = scale
        if scale == 16 :

            mide_feats = n_feats // 16
            mido_feats = 48
            last_feats = 3
            self.tail_13 = wn(nn.Conv2d(n_feats, n_feats, 3, padding=3//2, dilation=1))
            self.tail_15 = wn(nn.Conv2d(n_feats, n_feats, 5, padding=5//2, dilation=1))
            self.tail_17 = wn(nn.Conv2d(n_feats, n_feats, 7, padding=7//2, dilation=1))
            self.tail_19 = wn(nn.Conv2d(n_feats, n_feats, 9, padding=9//2, dilation=1))
        
            self.TFF1 = nn.Sequential(*[
                wn(nn.Conv2d(mide_feats*4, mide_feats, 1, padding=0, stride=1),),
            ])
            self.conv1 = wn(nn.Conv2d(mide_feats, mido_feats, 3, padding=3//2, dilation=1))
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.wdb_last = WDB_B(mido_feats, 3, mido_feats*3, wn, alpha=1.0,beta=1.0, act=self.act)

            self.tail_23 = wn(nn.Conv2d(mido_feats, mido_feats, 3, padding=3//2, dilation=1))
            self.tail_25 = wn(nn.Conv2d(mido_feats, mido_feats, 5, padding=5//2, dilation=1))
            self.tail_27 = wn(nn.Conv2d(mido_feats, mido_feats, 7, padding=7//2, dilation=1))
            self.tail_29 = wn(nn.Conv2d(mido_feats, mido_feats, 9, padding=9//2, dilation=1))
        
            self.TFF2 = nn.Sequential(*[
                wn(nn.Conv2d(last_feats*4, last_feats, 1, padding=0, stride=1),),
            ])

            self.pixelshuffle = nn.PixelShuffle(4)
        else :
            print('please check scale ',scale)
        
        
    def forward(self, x):
        if self.scale == 16 :
            outmix1 = []
            outmix1.append(self.pixelshuffle(self.tail_13(x)))
            outmix1.append(self.pixelshuffle(self.tail_15(x)))
            outmix1.append(self.pixelshuffle(self.tail_17(x)))
            outmix1.append(self.pixelshuffle(self.tail_19(x)))
        
            x = self.TFF1(torch.cat(outmix1,1))

            x = self.act(self.conv1(x))
            x = self.wdb_last(x)

            outmix2 = []
            outmix2.append(self.pixelshuffle(self.tail_23(x)))
            outmix2.append(self.pixelshuffle(self.tail_25(x)))
            outmix2.append(self.pixelshuffle(self.tail_27(x)))
            outmix2.append(self.pixelshuffle(self.tail_29(x)))
        
            x = self.TFF2(torch.cat(outmix2,1))
            
            return x
        else:
            print('!!!!! scale=',self.scale)
            pass

class ADC16_12(nn.Module):
    def __init__(self, args):
        super(ADC16_12, self).__init__()
        self.args = args
        self.n_resblocks = n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        block_feats = args.block_feats
        kernel_size = 3
        if args.act == 'relu' or 'RELU' or 'Relu':
            act = nn.ReLU(True)
        elif args.act == 'leakyrelu' or 'Leakyrelu' or 'LeakyReLU' : 
            act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        weight_init = 1 # don't need
        
        scale = args.scale[0] #16
        conv=common.default_conv

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [args.r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1]) #like WDSR

        # Shallow feature extraction net
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        self.WRDBs = nn.ModuleList()
        for _ in range(n_resblocks):
            self.WRDBs.append(
                WDN_B(n_feats, kernel_size, block_feats, wn, alpha=args.alpha,beta=args.beta, act=act,weight_init=weight_init)
            )

        self.GFF = nn.Sequential(*[
            wn(nn.Conv2d(n_resblocks * n_feats, n_feats, 1, padding=0, stride=1),),
            wn(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size-1)//2, stride=1))
        ])

        # define tail module

        tail = FFSC(args, scale, n_feats, kernel_size, wn,weight_init=weight_init) #AFSL

        skip = []
        for _ in range(int(math.log(scale, 4))):
                skip.append(
                    wn(nn.Conv2d(args.n_colors, args.n_colors * 16, 5, padding=5//2)))
                skip.append(nn.PixelShuffle(4))


        self.head = nn.Sequential(*m_head)
        self.tail = tail
        self.skip = nn.Sequential(*skip)

        a01,a12,a23,a34 = args.alpha,args.alpha,args.alpha,args.alpha
        b01,b12,b23,b34 = args.beta,args.beta,args.beta,args.beta
        self.A01 = Scale(a01)
        self.A12 = Scale(a12)
        self.A23 = Scale(a23)
        self.A34 = Scale(a34)
        self.A = [self.A01,self.A12,self.A23,self.A34]
        self.B01 = Scale(b01)
        self.B12 = Scale(b12)
        self.B23 = Scale(b23)
        self.B34 = Scale(b34)
        self.B = [self.B01,self.B12,self.B23,self.B34]

    def forward(self, x):
        if self.args.cpu:
            x = (x - self.rgb_mean*255)/127.5
        else :
            if self.args.precision == 'half':
                self.rgb_mean = self.rgb_mean.half()
            x = (x - self.rgb_mean.cuda()*255)/127.5 
        s = self.skip(x)
        x = self.head(x)
        WRDBs_out = []
        for i in range(self.n_resblocks):
            sk = x

            if i < self.n_resblocks-1:
                y = self.WRDBs[i](x)
                y = self.A[i](y)
                WRDBs_out.append(y)
                x = y + self.B[i](sk)
                
            elif i == self.n_resblocks-1:
                y = self.WRDBs[i](x)
                y = self.A[i](y)
                x = y + self.B[i](sk)
                WRDBs_out.append(x)

        x = self.GFF(torch.cat(WRDBs_out,1))
        x = self.tail(x)
        x += s
        
        if self.args.cpu:
            x = x*127.5 + self.rgb_mean*255
        else :
            x = x*127.5 + self.rgb_mean.cuda()*255
        return x

