import torch
import torch.nn as nn
from model_zoo.common import *
from model_zoo.ffc import *
from torchvision.ops import roi_align as torch_roi_align
import copy

class CFTL(nn.Module):
    def __init__(self,dim):
        super(CFTL, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.seq1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU()
        )

    def forward(self, x, xmask=None):

        N, C, H, W = x.size()
        # GlobalPooling
        x_g = self.avgpool(x).squeeze(-1).squeeze(-1)  # [N, C]
        x_g = x_g.unsqueeze(-1).unsqueeze(-1)  # [N, C, 1, 1]
        # CFFT
        x_fft = torch.fft.rfftn(x_g, dim=(-3, -2, -1))  # [N, C//2+1, 1, 1, 2]
        A, P = x_fft.abs(), x_fft.imag  # [N, C//2+1, 1, 1]
        # Process amplitude and phase
        A = self.seq1(x_g) * A
        P = self.seq2(x_g) * P
        # iCFFT
        x_fft = torch.complex(A * torch.cos(P), A * torch.sin(P))  # [N, C//2+1, 1, 1, 2]
        x_i = torch.fft.irfftn(x_fft, s=(C, 1, 1), dim=(-3, -2, -1))  # [N, C, 1, 1]
        # Repeat to original resolution
        y = x + x_i.repeat(1, 1, H, W)
        return y


class FFDG_network(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(FFDG_network, self).__init__()
        # self.conv_rgb1 = nn.Conv2d(in_channels=3, out_channels=num_feats,
        #                            kernel_size=kernel_size, padding=1)
        ffcconv_rgb = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(in_channels=3, out_channels=num_feats, kernel_size=7, padding=0,
                            ratio_gin=0, ratio_gout=0,
                            norm_layer=nn.BatchNorm2d,activation_layer=nn.ReLU, enable_lfu=False)]

        self.conv_rgb1 = nn.Sequential(*ffcconv_rgb)





        self.cat_up = ConcatTupleLayer()
        self.rgb_conv1 = FFC_BN_ACT(num_feats, num_feats, kernel_size=3, stride=1, padding=1,
                                    ratio_gin=0, ratio_gout=0.75, enable_lfu=False)
        self.rgb_rb1 = FFCResnetBlock(num_feats, num_feats, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)
        self.rgb_rb12 = FFCResnetBlock(num_feats, num_feats, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)
        self.rgb_rb13 = FFCResnetBlock(num_feats, num_feats, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)


        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg2 = ResidualGroup(default_conv, 2*num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg3 = ResidualGroup(default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg4 = ResidualGroup(default_conv, 4*num_feats, kernel_size, reduction=16, n_resblocks=4)

        self.bridge1 = SUFT(dp_feats=num_feats, add_feats=num_feats, scale=scale)
        self.bridge2 = SUFT(dp_feats=2*num_feats, add_feats=num_feats, scale=scale)
        self.bridge3 = SUFT(dp_feats=3*num_feats, add_feats=num_feats, scale=scale)

        my_tail = [
            ResidualGroup(
                default_conv, 4*num_feats, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(
                default_conv, 4*num_feats, kernel_size, reduction=16, n_resblocks=8)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(4*num_feats, 4*num_feats, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(4*num_feats, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        image, depth = x

        dp_in = self.act(self.conv_dp1(depth))
        dp1 = self.dp_rg1(dp_in)
        rgb1 = self.conv_rgb1(image)
        rgb0 = self.rgb_conv1(rgb1)
        frgb1 = self.rgb_rb1(rgb0)
        frgb2 = self.rgb_rb12(frgb1)
        frgb3 = self.rgb_rb13(frgb2)
        ca1_in = self.bridge1(dp1, self.cat_up(frgb1))
        dp2 = self.dp_rg2(ca1_in)
        ca2_in = self.bridge2(dp2, self.cat_up(frgb2))
        dp3 = self.dp_rg3(ca2_in)
        ca3_in = self.bridge3(dp3, self.cat_up(frgb3) )
        dp4 = self.dp_rg4(ca3_in)
        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))
        # hook_feats = self.bridge2.get_feats()
        return out

class FFDG_network2(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(FFDG_network2, self).__init__()
        # self.conv_rgb1 = nn.Conv2d(in_channels=3, out_channels=num_feats,
        #                            kernel_size=kernel_size, padding=1)
        ffcconv_rgb = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(in_channels=3, out_channels=num_feats, kernel_size=7, padding=0,
                            ratio_gin=0, ratio_gout=0,
                            norm_layer=nn.BatchNorm2d,activation_layer=nn.ReLU, enable_lfu=False)]
        self.conv_rgb1 = nn.Sequential(*ffcconv_rgb)
        self.grad = GCM(n_feats=num_feats,scale=scale)
        self.cat_up = ConcatTupleLayer()
        self.rgb_conv1 = FFC_BN_ACT(num_feats, num_feats, kernel_size=3, stride=1, padding=1,
                                    ratio_gin=0, ratio_gout=0.75, enable_lfu=False)
        self.rgb_rb1 = FFCResnetBlock(num_feats, num_feats, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)
        self.rgb_rb12 = FFCResnetBlock(num_feats, num_feats, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)
        self.rgb_rb13 = FFCResnetBlock(num_feats, num_feats, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)
        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        # self.conv_dp12 = nn.Conv2d(in_channels=1, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        # self.dp_rg12 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=4)

        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg2 = ResidualGroup(default_conv, 2*num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg3 = ResidualGroup(default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg4 = ResidualGroup(default_conv, 4*num_feats, kernel_size, reduction=16, n_resblocks=4)

        self.bridge1 = SUFT2(dp_feats=num_feats, add_feats=num_feats, scale=scale)
        self.bridge2 = SUFT2(dp_feats=2*num_feats, add_feats=num_feats, scale=scale)
        self.bridge3 = SUFT2(dp_feats=3*num_feats, add_feats=num_feats, scale=scale)

        my_tail = [
            ResidualGroup(
                default_conv, 4*num_feats, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(
                default_conv, 4*num_feats, kernel_size, reduction=16, n_resblocks=8)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(4*num_feats, 4*num_feats, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(4*num_feats, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = scale

    def forward(self, x):
        image, depth = x

        grad_attention = self.grad(depth,image)

        dp_in = self.act(self.conv_dp1(depth))
        dp1 = self.dp_rg1(dp_in)

        # dp_in2 = self.act2(self.conv_dp12(whole_depth_roi_pred))
        # dp12 = self.dp_rg12(dp_in2)

        rgb1 = self.conv_rgb1(image)
        rgb0 = self.rgb_conv1(rgb1)

        frgb1 = self.rgb_rb1(rgb0)
        frgb2 = self.rgb_rb12(frgb1)
        frgb3 = self.rgb_rb13(frgb2)
        # ca1_in = self.bridge1(dp1, dp12, self.cat_up(frgb1))
        ca1_in = self.bridge1(dp1,  self.cat_up(frgb1), grad_attention)
        dp2 = self.dp_rg2(ca1_in)

        ca2_in = self.bridge2(dp2, self.cat_up(frgb2), grad_attention)
        dp3 = self.dp_rg3(ca2_in)

        ca3_in = self.bridge3(dp3, self.cat_up(frgb3), grad_attention)
        dp4 = self.dp_rg4(ca3_in)

        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))
        # out = out + self.bicubic(depth)
        return out

class FFDG3_network(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(FFDG3_network, self).__init__()
        ffcconv_rgb = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(in_channels=3, out_channels=num_feats, kernel_size=7, padding=0,
                            ratio_gin=0, ratio_gout=0,
                            norm_layer=nn.BatchNorm2d,activation_layer=nn.ReLU, enable_lfu=False)]

        self.conv_rgb1 = nn.Sequential(*ffcconv_rgb)


        self.ffc_down1 =FFC_BN_ACT(num_feats, num_feats * 2, kernel_size=3, stride=2, padding=1,
                                ratio_gin=0, ratio_gout=0.75,
                                activation_layer=nn.ReLU, enable_lfu=False)
        self.ffc_down2 =FFC_BN_ACT(num_feats *2, num_feats * 4, kernel_size=3, stride=2, padding=1,
                                ratio_gin=0.75, ratio_gout=0.75,
                                activation_layer=nn.ReLU, enable_lfu=False)

        self.cat_up = ConcatTupleLayer()
        self.rgb_conv1 = FFC_BN_ACT(num_feats, num_feats, kernel_size=3, stride=1, padding=1,
                                    ratio_gin=0, ratio_gout=0.75, enable_lfu=False)
        self.rgb_rb1 = FFCResnetBlock(num_feats, num_feats, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)
        self.rgb_rb12 = FFCResnetBlock(num_feats, num_feats, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)
        self.rgb_rb13 = FFCResnetBlock(num_feats, num_feats, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)

        self.rgb_rb2 = FFCResnetBlock(num_feats * 2, num_feats * 2, ratio_gin=0.75, ratio_gout=0.75,  enable_lfu=False, inline=False)
        self.rgb_rb22 = FFCResnetBlock(num_feats * 2, num_feats * 2, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)
        self.rgb_rb23 = FFCResnetBlock(num_feats * 2, num_feats * 2, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)

        self.rgb_rb3 = FFCResnetBlock(num_feats * 4, num_feats * 4, ratio_gin=0.75, ratio_gout=0.75,  enable_lfu=False, inline=False)
        self.rgb_rb32 = FFCResnetBlock(num_feats * 4, num_feats * 4, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)
        self.rgb_rb33 = FFCResnetBlock(num_feats * 4, num_feats * 4, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False, inline=False)

        self.rgb_rb1up = DenseProjection(num_feats, num_feats, 1, up=False, bottleneck=False)
        self.rgb_rb2up = DenseProjection(num_feats*2, num_feats, 2, up=True, bottleneck=False)
        self.rgb_rb3up = DenseProjection(num_feats*4, num_feats, 4, up=True, bottleneck=False)


        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg2 = ResidualGroup(default_conv, 2*num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg3 = ResidualGroup(default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg4 = ResidualGroup(default_conv, 4*num_feats, kernel_size, reduction=16, n_resblocks=4)

        self.bridge1 = SUFT(dp_feats=num_feats, add_feats=num_feats, scale=scale)
        self.bridge2 = SUFT(dp_feats=2*num_feats, add_feats=num_feats, scale=scale)
        self.bridge3 = SUFT(dp_feats=3*num_feats, add_feats=num_feats, scale=scale)

        my_tail = [
            ResidualGroup(
                default_conv, 4*num_feats, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(
                default_conv, 4*num_feats, kernel_size, reduction=16, n_resblocks=8)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(4*num_feats, 4*num_feats, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(4*num_feats, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        image, depth = x

        dp_in = self.act(self.conv_dp1(depth))
        dp1 = self.dp_rg1(dp_in)

        rgb1 = self.conv_rgb1(image)
        rgb0 = self.rgb_conv1(rgb1)
        rgb2 = self.ffc_down1(rgb1)
        rgb3 = self.ffc_down2(rgb2)

        frgb1 = self.cat_up(self.rgb_rb13(self.rgb_rb12(self.rgb_rb1(rgb0))))
        frgb2 = self.cat_up(self.rgb_rb23(self.rgb_rb22(self.rgb_rb2(rgb2))))
        frgb3 = self.cat_up(self.rgb_rb33(self.rgb_rb32(self.rgb_rb3(rgb3))))

        frgb1up = self.rgb_rb1up(frgb1)
        frgb2up = self.rgb_rb2up(frgb2)
        frgb3up = self.rgb_rb3up(frgb3)

        ca1_in = self.bridge1(dp1, frgb1up)
        dp2 = self.dp_rg2(ca1_in)

        ca2_in = self.bridge2(dp2, frgb2up)
        dp3 = self.dp_rg3(ca2_in)

        ca3_in = self.bridge3(dp3, frgb3up)
        dp4 = self.dp_rg4(ca3_in)

        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))

        # out = out + self.bicubic(depth)

        return out

