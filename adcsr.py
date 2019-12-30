import torch
import torch.nn as nn
import math
from model import common


def make_model(args, parent=False):
    return ADCSR(args)

def abandon():
    return ['tail']

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


class CONVU(nn.Module):
    def __init__(
        self, n_feats, kernel_size, block_feats, wn,  act=nn.ReLU(True),weight_init=1):
        super(CONVU, self).__init__()
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

class ADRB(nn.Module):
    def __init__(
        self, n_feats, kernel_size, block_feats, wn, alpha=1.0,beta=1.0, act=nn.ReLU(True), weight_init=1):
        super(ADRB, self).__init__()
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

        self.get_Y1=CONVU(n_feats,kernel_size,block_feats,wn,act=act,weight_init=weight_init)
        self.get_Y2=CONVU(n_feats,kernel_size,block_feats,wn,act=act,weight_init=weight_init)
        self.get_Y3=CONVU(n_feats,kernel_size,block_feats,wn,act=act,weight_init=weight_init)
        self.get_Y4=CONVU(n_feats,kernel_size,block_feats,wn,act=act,weight_init=weight_init)
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

class ADRU(nn.Module):
    def __init__(self, n_feats, kernel_size, block_feats, wn, alpha=1.0,beta=1.0, act=nn.ReLU(True),weight_init=1):
        super(ADRU, self).__init__()

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


        self.wdb1=ADRB(n_feats, kernel_size, block_feats, wn, alpha, beta,act=act,weight_init=weight_init)
        self.wdb2=ADRB(n_feats, kernel_size, block_feats, wn, alpha, beta,act=act,weight_init=weight_init)
        self.wdb3=ADRB(n_feats, kernel_size, block_feats, wn, alpha, beta,act=act,weight_init=weight_init)
        self.wdb4=ADRB(n_feats, kernel_size, block_feats, wn, alpha, beta,act=act,weight_init=weight_init)
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
    

class AFSC(nn.Module): #AFSL
    def __init__(
        self, args, scale, n_feats, kernel_size, wn, weight_init=1):
        super(AFSC, self).__init__()
        self.scale = scale
        if scale in [2,3,4,8]:
            lastfeats = 3
            out_feats = scale*scale*lastfeats
            
            self.tail_3 = wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2, dilation=1))
            self.tail_5 = wn(nn.Conv2d(n_feats, out_feats, 5, padding=5//2, dilation=1))
            self.tail_7 = wn(nn.Conv2d(n_feats, out_feats, 7, padding=7//2, dilation=1))
            self.tail_9 = wn(nn.Conv2d(n_feats, out_feats, 9, padding=9//2, dilation=1))
        
            self.TFF = nn.Sequential(*[
                wn(nn.Conv2d(lastfeats*4, lastfeats, 1, padding=0, stride=1),),
            ])
            self.pixelshuffle = nn.PixelShuffle(scale)
        else :
            print('please check scale ',scale)
        
        
    def forward(self, x):
        if self.scale in [2,3,4,8]:
            outmix = []
            outmix.append(self.pixelshuffle(self.tail_3(x)))
            outmix.append(self.pixelshuffle(self.tail_5(x)))
            outmix.append(self.pixelshuffle(self.tail_7(x)))
            outmix.append(self.pixelshuffle(self.tail_9(x)))
        
            x = self.TFF(torch.cat(outmix,1))
            return x
        else:
            pass

class ADCSR(nn.Module):
    def __init__(self, args):
        super(ADCSR, self).__init__()
        self.args = args
        self.n_resblocks = n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        block_feats = args.block_feats
        kernel_size = 3
        if args.act == 'relu' or 'RELU' or 'Relu':
            act = nn.ReLU(True)
        elif args.act == 'leakyrelu' or 'Leakyrelu' or 'LeakyReLU' :
            act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        weight_init = 1.0
        
        scale = args.scale[0] #2 3 4 8
        conv=common.default_conv

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [args.r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1])

        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        
        self.ADRUs = nn.ModuleList()
        for _ in range(n_resblocks):
            self.ADRUs.append(
                ADRU(n_feats, kernel_size, block_feats, wn, alpha=args.alpha,beta=args.beta, act=act,weight_init=weight_init)
            )

        self.GFF = nn.Sequential(*[
            wn(nn.Conv2d(n_resblocks * n_feats, n_feats, 1, padding=0, stride=1),),
            wn(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size-1)//2, stride=1))
        ])

        tail = AFSC(args, scale, n_feats, kernel_size, wn,weight_init=weight_init)

        skip = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                skip.append(
                    wn(nn.Conv2d(args.n_colors, args.n_colors * 4, 5, padding=5//2)))
                skip.append(nn.PixelShuffle(2))
        elif scale == 3:
                skip.append(
                    wn(nn.Conv2d(args.n_colors, args.n_colors * 9, 5, padding=5//2)))
                skip.append(nn.PixelShuffle(3))


        self.head = nn.Sequential(*m_head)
        self.tail = tail
        # self.tail = nn.Sequential(*tail)
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
        ADRUs_out = []
        for i in range(self.n_resblocks):
            sk = x

            if i < self.n_resblocks-1:
                y = self.ADRUs[i](x)
                y = self.A[i](y)
                ADRUs_out.append(y)
                x = y + self.B[i](sk)
                
            elif i == self.n_resblocks-1:
                y = self.ADRUs[i](x)
                y = self.A[i](y)
                x = y + self.B[i](sk)
                ADRUs_out.append(x)

        x = self.GFF(torch.cat(ADRUs_out,1))
        x = self.tail(x)
        x += s
        
        if self.args.cpu:
            x = x*127.5 + self.rgb_mean*255
        else :
            x = x*127.5 + self.rgb_mean.cuda()*255
        return x

