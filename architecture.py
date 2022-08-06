import torch
import torch.nn as nn
from einops import rearrange
from globals import RGB_img_res


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device, stride=1, depth=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels * depth,
                                   kernel_size=kernel_size,
                                   groups=depth,
                                   padding=1,
                                   stride=stride,
                                   bias=bias).to(device)
        self.pointwise = nn.Conv2d(out_channels * depth, out_channels, kernel_size=(1, 1), bias=bias).to(device)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        SeparableConv2d(in_channels=inp, out_channels=oup, kernel_size=kernal_size, stride=stride,
                        bias=False, device='cuda:0'),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(), 
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(), 
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout) 

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [1, 1, 1]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion)) 
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])


    def forward(self, x):
        y0 = self.conv1(x)
        x = self.mv2[0](y0)

        y1 = self.mv2[1](x)
        x = self.mv2[2](y1)
        x = self.mv2[3](x)  # Repeat

        y2 = self.mv2[4](x)
        x = self.mvit[0](y2)

        y3 = self.mv2[5](x)
        x = self.mvit[1](y3)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        return x, [y0, y1, y2, y3]


def mobilevit_xxs():
    enc_type = 'xxs'
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 160]  # 320
    return MobileViT((RGB_img_res[1], RGB_img_res[2]), dims, channels, expansion=2), enc_type


def mobilevit_xs():
    enc_type = 'xs'
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 192] # 384
    return MobileViT((RGB_img_res[1], RGB_img_res[2])), dims, channels), enc_type


def mobilevit_s():
    enc_type = 's'
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 320]
    return MobileViT((RGB_img_res[1], RGB_img_res[2]), dims, channels), enc_type


class UpSample_layer(nn.Module):
    def __init__(self, inp, oup, flag, sep_conv_filters, name, device):
        super(UpSample_layer, self).__init__()
        self.flag = flag
        self.name = name
        self.conv2d_transpose = nn.ConvTranspose2d(inp, oup, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                   dilation=1, output_padding=(1, 1), bias=False)
        self.end_up_layer = nn.Sequential(
            SeparableConv2d(sep_conv_filters, oup, kernel_size=(3, 3), device=device),
            nn.ReLU()
        )


    def forward(self, x, enc_layer):
        x = self.conv2d_transpose(x)
        if x.shape[-1] != enc_layer.shape[-1]:
            enc_layer = torch.nn.functional.pad(enc_layer, pad=(1, 0), mode='constant', value=0.0)
        if x.shape[-1] != enc_layer.shape[-1]:
            enc_layer = torch.nn.functional.pad(enc_layer, pad=(0, 1), mode='constant', value=0.0)
        x = torch.cat([x, enc_layer], dim=1)
        x = self.end_up_layer(x)

        return x


class decoder(nn.Module):
    def __init__(self, device, typ):
        super(decoder, self).__init__()
        self.conv2d_in = nn.Conv2d(320 if typ == 's' else 192 if typ == 'xs' else 160,
                                   128 if typ == 's' else 128 if typ == 'xs' else 64,
                                   kernel_size=(1, 1), padding='same', bias=False)
        self.ups_block_1 = UpSample_layer(128 if typ == 's' else 128 if typ == 'xs' else 64,
                                          64 if typ == 's' else 64 if typ == 'xs' else 32,
                                          flag=True,
                                          sep_conv_filters=192 if typ == 's' else 144 if typ == 'xs' else 96,
                                          name='up1', device=device)
        self.ups_block_2 = UpSample_layer(64 if typ == 's' else 64 if typ == 'xs' else 32,
                                          32 if typ == 's' else 32 if typ == 'xs' else 16,
                                          flag=False,
                                          sep_conv_filters=128 if typ == 's' else 96 if typ == 'xs' else 64,
                                          name='up2', device=device)
        self.ups_block_3 = UpSample_layer(32 if typ == 's' else 32 if typ == 'xs' else 16,
                                          16 if typ == 's' else 16 if typ == 'xs' else 8,
                                          flag=False,
                                          sep_conv_filters=80 if typ == 's' else 64 if typ == 'xs' else 32,
                                          name='up3', device=device)
        self.conv2d_out = nn.Conv2d(16 if typ == 's' else 16 if typ == 'xs' else 8,
                                    1, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, x, enc_layer_list):
        x = self.conv2d_in(x)
        x = self.ups_block_1(x, enc_layer_list[3])
        x = self.ups_block_2(x, enc_layer_list[2])
        x = self.ups_block_3(x, enc_layer_list[1])
        x = self.conv2d_out(x)
        return x


class build_METER_model(nn.Module):
    def __init__(self, device, arch_type):
        super(build_METER_model, self).__init__()
        if arch_type == 's':
            self.encoder, enc_type = mobilevit_s()
        elif arch_type == 'xs':
            self.encoder, enc_type = mobilevit_xs()
        else:
            self.encoder, enc_type = mobilevit_xxs()
        self.decoder = decoder(device=device, typ=enc_type)

    def forward(self, x):
        x, enc_layer = self.encoder(x)
        x = self.decoder(x, enc_layer)
        return x