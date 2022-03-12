import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.project =nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += self.project(residual)
        return F.relu(x)


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()

        self.block = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
        # self.pool = nn.Conv2d(out_ch, out_ch, 3,stride=2,padding=1)

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        return x


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.upsample(x)
        x = self.block(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class ResidualUNet2D_affs(nn.Module):
    def __init__(self, in_channels=3,
                out_channels=3,
                nfeatures=[16,32,64,128,256],
                if_sigmoid=True,
                show_feature=False):
        super(ResidualUNet2D_affs, self).__init__()
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        self.inconv = InConv(in_channels, nfeatures[0])
        self.down1 = Down(nfeatures[0], nfeatures[1])
        self.down2 = Down(nfeatures[1], nfeatures[2])
        self.down3 = Down(nfeatures[2], nfeatures[3])
        self.down4 = Down(nfeatures[3], nfeatures[4])

        self.up1_emb = Up(nfeatures[4], nfeatures[4])
        self.up2_emb = Up(nfeatures[4]+nfeatures[3], nfeatures[3])
        self.up3_emb = Up(nfeatures[3]+nfeatures[2], nfeatures[2])
        self.up4_emb = Up(nfeatures[2]+nfeatures[1], nfeatures[1])
        # self.outconv_emb = OutConv(nfeatures[1], n_emb)

        self.binary_seg = nn.Sequential(
            nn.Conv2d(nfeatures[1], nfeatures[1], 1),
            nn.BatchNorm2d(nfeatures[1]),
            nn.ReLU(),
            nn.Conv2d(nfeatures[1], out_channels, 1)
        )

    def concat_channels(self, x_cur, x_prev):
        if x_cur.shape!=x_prev.shape:
            p1 = x_prev.shape[-1] - x_cur.shape[-1]
            p2 = x_prev.shape[-2] - x_cur.shape[-2]
            padding = nn.ReplicationPad2d((0, p2, 0, p1)).cuda()
            x_cur = padding(x_cur)
        return torch.cat([x_cur, x_prev], dim=1)

    def forward(self, x):
        #encoder
        x = self.inconv(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        x_emb0 = self.up1_emb(x)
        x_emb0 = self.concat_channels(x_emb0, x4)
        x_emb0 = self.up2_emb(x_emb0)
        x_emb0 = self.concat_channels(x_emb0, x3)
        x_emb0 = self.up3_emb(x_emb0)
        x_emb0 = self.concat_channels(x_emb0, x2)
        x_emb0 = self.up4_emb(x_emb0)
        # x_emb = self.outconv_emb(x_emb0)
        binary_seg = self.binary_seg(x_emb0)

        if self.if_sigmoid:
            binary_seg = torch.sigmoid(binary_seg)

        if self.show_feature:
            return x_emb0, binary_seg
        else:
            return binary_seg


class ResidualUNet2D_affs2(nn.Module):
    def __init__(self, in_channels=3,
                out_channels=3,
                nfeatures=[16,32,64,128,256],
                if_sigmoid=True,
                show_feature=False):
        super(ResidualUNet2D_affs2, self).__init__()
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        self.inconv = InConv(in_channels, nfeatures[0])
        self.down1 = Down(nfeatures[0], nfeatures[1])
        self.down2 = Down(nfeatures[1], nfeatures[2])
        self.down3 = Down(nfeatures[2], nfeatures[3])
        self.down4 = Down(nfeatures[3], nfeatures[4])

        self.up1_emb = Up(nfeatures[4], nfeatures[4])
        self.up2_emb = Up(nfeatures[4]+nfeatures[3], nfeatures[3])
        self.up3_emb = Up(nfeatures[3]+nfeatures[2], nfeatures[2])
        self.up4_emb = Up(nfeatures[2]+nfeatures[1], nfeatures[1])
        # self.outconv_emb = OutConv(nfeatures[1], n_emb)

        self.binary_seg = nn.Sequential(
            nn.Conv2d(nfeatures[1], nfeatures[1], 1),
            nn.BatchNorm2d(nfeatures[1]),
            nn.ReLU(),
            nn.Conv2d(nfeatures[1], 2, 1)
        )

        # self.out_affs = nn.Sequential(
        #     nn.Conv2d(nfeatures[1], nfeatures[1], 1),
        #     nn.BatchNorm2d(nfeatures[1]),
        #     nn.ReLU(),
        #     nn.Conv2d(nfeatures[1], out_channels, 1)
        # )
        self.out_affs = OutConv(nfeatures[1], out_channels)

    def concat_channels(self, x_cur, x_prev):
        if x_cur.shape!=x_prev.shape:
            p1 = x_prev.shape[-1] - x_cur.shape[-1]
            p2 = x_prev.shape[-2] - x_cur.shape[-2]
            padding = nn.ReplicationPad2d((0, p2, 0, p1)).cuda()
            x_cur = padding(x_cur)
        return torch.cat([x_cur, x_prev], dim=1)

    def forward(self, x):
        #encoder
        x = self.inconv(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        x_emb0 = self.up1_emb(x)
        x_emb0 = self.concat_channels(x_emb0, x4)
        x_emb0 = self.up2_emb(x_emb0)
        x_emb0 = self.concat_channels(x_emb0, x3)
        x_emb0 = self.up3_emb(x_emb0)
        x_emb0 = self.concat_channels(x_emb0, x2)
        x_emb0 = self.up4_emb(x_emb0)
        # x_emb = self.outconv_emb(x_emb0)
        out_affs = self.out_affs(x_emb0)
        binary_seg = self.binary_seg(x_emb0)

        if self.if_sigmoid:
            out_affs = torch.sigmoid(out_affs)

        if self.show_feature:
            return x_emb0, out_affs, binary_seg
        else:
            return out_affs, binary_seg


class ResidualUNet2D_embedding(nn.Module):
    def __init__(self, in_channels=3,
                out_channels=2,
                nfeatures=[16,32,64,128,256],
                emd=16,
                if_sigmoid=False,
                show_feature=False):
        super(ResidualUNet2D_embedding, self).__init__()
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        self.inconv = InConv(in_channels, nfeatures[0])
        self.down1 = Down(nfeatures[0], nfeatures[1])
        self.down2 = Down(nfeatures[1], nfeatures[2])
        self.down3 = Down(nfeatures[2], nfeatures[3])
        self.down4 = Down(nfeatures[3], nfeatures[4])

        self.up1_emb = Up(nfeatures[4], nfeatures[4])
        self.up2_emb = Up(nfeatures[4]+nfeatures[3], nfeatures[3])
        self.up3_emb = Up(nfeatures[3]+nfeatures[2], nfeatures[2])
        self.up4_emb = Up(nfeatures[2]+nfeatures[1], nfeatures[1])
        self.outconv_emb = OutConv(nfeatures[1], emd)

        self.binary_seg = nn.Sequential(
            nn.Conv2d(nfeatures[1], nfeatures[1], 1),
            nn.BatchNorm2d(nfeatures[1]),
            nn.ReLU(),
            nn.Conv2d(nfeatures[1], out_channels, 1)
        )

    def concat_channels(self, x_cur, x_prev):
        if x_cur.shape!=x_prev.shape:
            p1 = x_prev.shape[-1] - x_cur.shape[-1]
            p2 = x_prev.shape[-2] - x_cur.shape[-2]
            padding = nn.ReplicationPad2d((0, p2, 0, p1)).cuda()
            x_cur = padding(x_cur)
        return torch.cat([x_cur, x_prev], dim=1)

    def forward(self, x):
        #encoder
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_emb1 = self.up1_emb(x5)
        x_emb2 = self.concat_channels(x_emb1, x4)
        x_emb2 = self.up2_emb(x_emb2)
        x_emb3 = self.concat_channels(x_emb2, x3)
        x_emb3 = self.up3_emb(x_emb3)
        x_emb4 = self.concat_channels(x_emb3, x2)
        x_emb4 = self.up4_emb(x_emb4)
        x_emb = self.outconv_emb(x_emb4)
        binary_seg = self.binary_seg(x_emb4)

        # if self.if_sigmoid:
        #     binary_seg = torch.sigmoid(binary_seg)

        if self.show_feature:
            return x5, x_emb1, x_emb2, x_emb3, x_emb, binary_seg
        else:
            return x_emb, binary_seg


class ResidualUNet2D_deep(nn.Module):
    def __init__(self, in_channels=3,
                out_channels=2,
                nfeatures=[16,32,64,128,256],
                emd=16,
                if_sigmoid=False,
                show_feature=False):
        super(ResidualUNet2D_deep, self).__init__()
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        self.inconv = InConv(in_channels, nfeatures[0])
        self.down1 = Down(nfeatures[0], nfeatures[1])
        self.down2 = Down(nfeatures[1], nfeatures[2])
        self.down3 = Down(nfeatures[2], nfeatures[3])
        self.down4 = Down(nfeatures[3], nfeatures[4])

        self.up1_emb = Up(nfeatures[4], nfeatures[4])
        self.up2_emb = Up(nfeatures[4]+nfeatures[3], nfeatures[3])
        self.up3_emb = Up(nfeatures[3]+nfeatures[2], nfeatures[2])
        self.up4_emb = Up(nfeatures[2]+nfeatures[1], nfeatures[1])
        self.outconv1 = OutConv(nfeatures[4], emd)
        # self.outconv2 = OutConv(nfeatures[4]+nfeatures[3], emd)
        # self.outconv3 = OutConv(nfeatures[3]+nfeatures[2], emd)
        # self.outconv4 = OutConv(nfeatures[2]+nfeatures[1], emd)
        self.outconv2 = OutConv(nfeatures[4], emd)
        self.outconv3 = OutConv(nfeatures[3], emd)
        self.outconv4 = OutConv(nfeatures[2], emd)
        self.outconv_emb = OutConv(nfeatures[1], emd)

        self.binary_seg = nn.Sequential(
            nn.Conv2d(nfeatures[1], nfeatures[1], 1),
            nn.BatchNorm2d(nfeatures[1]),
            nn.ReLU(),
            nn.Conv2d(nfeatures[1], out_channels, 1)
        )

    def concat_channels(self, x_cur, x_prev):
        if x_cur.shape!=x_prev.shape:
            p1 = x_prev.shape[-1] - x_cur.shape[-1]
            p2 = x_prev.shape[-2] - x_cur.shape[-2]
            padding = nn.ReplicationPad2d((0, p2, 0, p1)).cuda()
            x_cur = padding(x_cur)
        return torch.cat([x_cur, x_prev], dim=1)

    def forward(self, x):
        #encoder
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_emb1 = self.outconv1(x5)

        x_emb0 = self.up1_emb(x5)
        x_emb2 = self.outconv2(x_emb0)

        x_emb0 = self.concat_channels(x_emb0, x4)
        x_emb0 = self.up2_emb(x_emb0)
        x_emb3 = self.outconv3(x_emb0)

        x_emb0 = self.concat_channels(x_emb0, x3)
        x_emb0 = self.up3_emb(x_emb0)
        x_emb4 = self.outconv4(x_emb0)

        x_emb0 = self.concat_channels(x_emb0, x2)
        x_emb0 = self.up4_emb(x_emb0)
        x_emb5 = self.outconv_emb(x_emb0)

        binary_seg = self.binary_seg(x_emb0)

        # if self.if_sigmoid:
        #     binary_seg = torch.sigmoid(binary_seg)

        return x_emb1, x_emb2, x_emb3, x_emb4, x_emb5, binary_seg


if __name__ == '__main__':
    import numpy as np
    from ptflops import get_model_complexity_info

    x = torch.Tensor(np.random.random((4, 3, 448, 448)).astype(np.float32)).cuda()
    # model = ResidualUNet2D_deep(out_channels=2).cuda()
    # model = ResidualUNet2D_embedding().cuda()
    # model = ResidualUNet2D_embedding(nfeatures=[64,128,256,512,1024]).cuda()  # 1254.23, 75.15M
    # model = ResidualUNet2D_embedding(nfeatures=[32,64,128,256,512]).cuda()  # 314.39, 18.8M
    # model = ResidualUNet2D_embedding(nfeatures=[16,32,64,128,256]).cuda()  # 79.01, 4.7M
    # model = ResidualUNet2D_embedding(nfeatures=[16,32,48,64,128]).cuda()  # 40.59, 1.34M
    # model = ResidualUNet2D_embedding(nfeatures=[16,20,30,40,64]).cuda()  # 16.5, 421.5k
    # model = ResidualUNet2D_embedding(nfeatures=[8,12,16,20,32]).cuda()  # 5.15, 111.8k
    # model = ResidualUNet2D_embedding(nfeatures=[4,8,10,12,16]).cuda()  # 2.08, 36.47k
    model = ResidualUNet2D_deep().cuda()

    emb1, emb2, emb3, emb4, emb, mask = model(x)
    # emb, mask = model(x)
    # print(emb1.shape, emb2.shape, emb3.shape, emb4.shape, emb.shape)
    print(emb.shape)
    print(mask.shape)

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))