import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch,ks=3,s=1,pad=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=s, padding=pad, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch,ks=3,s=1,pad=1,op=0):
        super(up_conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=ks, stride=s, output_padding=op,padding=pad, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# class Former(nn.Module):
#     def __init__(self, in_ch=3):
#         super(Former, self).__init__()
#         self.conv0=conv_block(in_ch,32,ks=1,pad=0)
#         self.conv1 = block(32, 32)
#         self.conv2=block(32,64)
#         self.conv3 = block(64, 32)
#
#         self.conv_last = nn.Sequential(
#             nn.Conv2d(32,1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.Sigmoid()
#             )
#
#     def forward(self, x):
#         x=self.conv0(x)
#         x=self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         out=self.conv_last(x)
#         return out

class Former(nn.Module):
    def __init__(self, in_ch=3):
        super(Former, self).__init__()
        self.conv0=conv_block(in_ch,32,ks=1,pad=0)
        self.conv1 = block(32, 32)
        self.conv2=block(32,64)
        self.conv3 = block(64, 128)
        self.conv4 = block(128, 64)
        self.conv5 = block(64,32)

        self.conv_last = nn.Sequential(
            nn.Conv2d(32,1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
            )

        # self.conv_last = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)
        # )

    def forward(self, x):
        x=self.conv0(x)
        x=self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out=self.conv_last(x)
        return out

class block(nn.Module):
    def __init__(self,in_channel,out_channel,ks=3,pad=1, head_num=4):
        super(block, self).__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.ma=MultiheadAttention(out_channel,out_channel, head_num)

        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm=nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self,x):
        x = self.conv1(x)
        mx=self.ma(x)
        x=x+mx
        x=self.norm(x)

        # cx=self.conv2(x)
        # x=x+cx
        # x = self.norm(x)
        return x

# class block(nn.Module):
#     def __init__(self,in_channel,out_channel,ks=3,pad=1, head_num=4):
#         super(block, self).__init__()
#         self.conv = conv_block(out_channel, out_channel, ks=ks, pad=pad)
#         self.ma=MultiheadAttention(in_channel,out_channel, head_num)
#
#     def forward(self,x):
#         x=self.ma(x)
#         cx=self.conv(x)
#         x=x+cx
#         return x

class MultiheadAttention(nn.Module):
    def __init__(self,in_channel,out_channel, head_num=4):
        super(MultiheadAttention, self).__init__()
        self.uf_ks=3
        # self.uf_ks = 5
        self.head_num=head_num
        self.ic=in_channel
        self.oc=out_channel

        self.qkv=nn.ModuleList([nn.Conv2d(in_channel, (out_channel//head_num)*3, kernel_size=1, padding=0, bias=False) for i in range(self.head_num)])

        self.cat=nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        #x: b,c,h,w
        b,_,h,w=x.shape

        cat = [net(x) for net in self.qkv]  # b,c,h,w
        qkv=[torch.split(y,self.oc//self.head_num,dim=1) for y in cat]
        query=[tmp[0] for tmp in qkv]
        key=[tmp[1] for tmp in qkv]
        value=[tmp[2] for tmp in qkv]


        u_key=[F.pad(k, pad=[self.uf_ks//2,self.uf_ks//2,self.uf_ks//2,self.uf_ks//2],mode='replicate') for k in key]#b,c,h+ks//2,w+ks//2
        u_key=[k.unfold(2,self.uf_ks,1).unfold(3,self.uf_ks,1) for k in u_key]#b,c,h,w,ks,ks

        u_value = [F.pad(v, pad=[self.uf_ks // 2, self.uf_ks // 2, self.uf_ks // 2, self.uf_ks // 2], mode='replicate') for v in value]  # b,c,h+ks//2,w+ks//2
        u_value = [v.unfold(2, self.uf_ks, 1).unfold(3, self.uf_ks, 1) for v in u_value]  # b,c,h,w,ks,ks

        query=[q.permute(0,2,3,1).unsqueeze(dim=3) for q in query]# b,h,w,1,c
        u_key=[k.permute(0,2,3,1,4,5).reshape(b,h,w,self.oc//self.head_num,self.uf_ks*self.uf_ks) for k in u_key]# b,h,w,c,ks*ks

        s=[torch.einsum('bhwik,bhwkj->bhwij',query[i],u_key[i]) for i in range(self.head_num)]#b,h,w,1,ks*ks
        s=[F.softmax(ss.squeeze(dim=-2), dim=-1) for ss in s]#b,h,w,ks*ks

        res=[s[i].unsqueeze(dim=1)*u_value[i].reshape(b,self.oc//self.head_num,h,w,self.uf_ks*self.uf_ks) for i in range(self.head_num)]#b,c,h,w,ks*ks
        res=[torch.sum(r,dim=-1) for r in res]#b,c,h,w
        res=torch.cat(res,dim=1)
        res=self.cat(res)

        return res