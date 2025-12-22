import copy
import math
import functools
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import spectral_norm
from src.common import BaseNetwork
import torch.nn.functional as F


class AdversarialLoss(nn.Module):

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class PerceptualLoss(nn.Module):

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class ChannelAttention(nn.Module):

    def __init__(self, channel, reduction_ratio=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)  # [B, C]
        y = self.fc(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        return x * y


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class Contrastive(nn.Module):
    def __init__(self, netD='Unet', no_mlp=True, input_dim=1024, mlp_dim=256, temp=0.07, edge_weight=0.5):
        super(Contrastive, self).__init__()
        if netD == 'Unet':
            self.netD = UnetDiscriminator().cuda()
        self.l1 = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.no_mlp = no_mlp
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.edge_weight = edge_weight  # 边缘损失权重
        self.margin = nn.Parameter(torch.tensor(1.0))

        # Sobel边缘检测卷积核初始化
        self.sobel_conv = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_kernel = self._init_sobel_kernel()
        self.sobel_conv.weight.data = sobel_kernel
        self.sobel_conv.requires_grad_(False)  # 固定参数

        if not no_mlp:
            self.mlp = nn.Sequential(
                nn.Conv2d(input_dim, mlp_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(mlp_dim, mlp_dim, kernel_size=1)
            )
            self.mlp.apply(weights_init)
            self.mlp.cuda()
        # self.temp = temp
        self.temp = nn.Parameter(torch.tensor(temp))  # 初始值0.07

    def _init_sobel_kernel(self):
        kernel = torch.tensor([
            [[[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
            [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]
        ], dtype=torch.float32).repeat(1, 3, 1, 1)
        return kernel / 8.0

    def compute_edge_features(self, x):
        """计算边缘特征图"""
        edge_maps = self.sobel_conv(x)  # [B,2,H,W]
        edge_magnitude = torch.sqrt(edge_maps[:, 0] ** 2 + edge_maps[:, 1] ** 2 + 1e-6)
        return edge_magnitude.unsqueeze(1)  # [B,1,H,W]

    def edge_aware_loss(self, pred, gt, mask):
        """
        pred:  [B,C,H,W]
        gt:  [B,C,H,W]
        mask: mask [B,1,H,W]，[0,1]
        """

        B, C, H, W = gt.shape
        mask = mask.to(gt.dtype).to(gt.device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B,1,H,W]


        masked_gt = gt * (1 - mask) + mask


        pred_edge = self.compute_edge_features(pred)
        gt_edge = self.compute_edge_features(gt)
        masked_gt_edge = self.compute_edge_features(masked_gt)


        residual_features_pred, _, _, _ = self.netD(pred)
        pred_feat = residual_features_pred[3]  # [B,C,h,w]

        residual_features_gt, _, _, _ = self.netD(gt)
        gt_feat = residual_features_gt[3]  # [B,C,h,w]


        masked_gt_edge_rgb = masked_gt_edge.repeat(1, 3, 1, 1)  # [B,3,H,W]
        residual_features_masked, _, _, _ = self.netD(masked_gt_edge_rgb)
        masked_gt_feat = residual_features_masked[3]  # [B,C,h,w]


        edge_mask = torch.sigmoid(5 * (gt_edge - 0.5))  # [B,1,H,W]
        edge_mask = F.interpolate(
            edge_mask,
            size=pred_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )  # [B,1,h,w]


        weighted_pred = pred_feat * edge_mask  # 生成图像特征加权
        weighted_gt = gt_feat * edge_mask      # 真实图像特征加权
        weighted_masked_gt = masked_gt_feat * edge_mask  # 掩码边缘特征加权（负样本）


        pos_sim = F.cosine_similarity(weighted_pred, weighted_gt, dim=1)  # 正样本相似度
        neg_sim = F.cosine_similarity(weighted_pred, weighted_masked_gt, dim=1)  # 负样本相似度

        loss = -torch.log(torch.exp(pos_sim / self.temp) / (torch.exp(pos_sim / self.temp) + torch.exp(neg_sim / self.temp)))
        return loss.mean()

    def reduce_normalize(self, feat):
        if not self.no_mlp:
            feat = self.mlp(feat)

        feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)  # [B, C]
        norm = feat.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = feat.div(norm + 1e-7)
        return out

    def __call__(self, input, predict_result, gt, mask, num_feat_layers):

        edge_loss = self.edge_aware_loss(predict_result, gt, mask) * self.edge_weight


        _, feat_output_final, _, _ = self.netD(predict_result)
        _, feat_gt_final, _, _ = self.netD(gt)
        _, feat_corrupted_final, _, _ = self.netD(input)


        feat_output_final = self.reduce_normalize(feat_output_final)  # [B, D]
        feat_gt_final = self.reduce_normalize(feat_gt_final)  # [B, D]
        feat_corrupted_final = self.reduce_normalize(feat_corrupted_final)  # [B, D]


        batch_size = input.size(0)


        pos_sim = torch.sum(feat_output_final * feat_gt_final, dim=1)  # [B]
        neg_sim = torch.sum(feat_output_final * feat_corrupted_final, dim=1)  # [B]


        margin = self.margin
        triplet_loss = torch.relu(neg_sim - pos_sim + margin)

        # 取批次平均损失
        semantic_loss = triplet_loss.mean()

        return edge_loss, semantic_loss


def weights_init(init_type='orthogonal'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


# ----- A simple U-net discriminator -----
class UnetDiscriminator(BaseNetwork):

    def __init__(self, D_ch=64,
                 D_attn='64', D_activation='relu',
                 output_dim=1, D_init='orthogonal', **kwargs):
        super(UnetDiscriminator, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Resolution
        self.resolution = 256
        # Attention?
        self.use_attn = False
        self.attention = D_attn
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.use_SN = not False

        self.save_features = [0, 1, 2, 3, 4, 5]

        self.out_channel_multiplier = 1  # 4
        # Architecture
        self.arch = D_unet_arch(self.ch, self.attention)[self.resolution]

        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[EBlock(inc=self.arch['in_channels'][index],
                                        outc=self.arch['out_channels'][index],
                                        use_SN=self.use_SN,
                                        activation=self.activation)]]

            elif self.arch["upsample"][index]:

                self.blocks += [[DBlock(inc=self.arch['in_channels'][index],
                                        outc=self.arch['out_channels'][index],
                                        use_SN=self.use_SN,
                                        activation=self.activation)]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][
                self.arch['resolution'][index]] and attention_condition and self.use_attn:  # index < 5
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.use_SN)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        last_layer = nn.Conv2d(self.ch * self.out_channel_multiplier, 1, kernel_size=1)
        self.blocks.append(last_layer)

        self.linear_middle = nn.Linear(16 * self.ch, output_dim)
        if self.use_SN:
            self.linear_middle = spectral_norm(self.linear_middle)

        self.init_weights(init_type=self.init)
        self.print_network()

    def forward(self, x):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):
            if index == 7:
                h = torch.cat((h, residual_features[5]), dim=1)
            elif index == 8:
                h = torch.cat((h, residual_features[4]), dim=1)
            elif index == 9:
                h = torch.cat((h, residual_features[3]), dim=1)
            elif index == 10:
                h = torch.cat((h, residual_features[2]), dim=1)
            elif index == 11:
                h = torch.cat((h, residual_features[1]), dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index == self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(get_non_linearity(self.activation)()(h), [2, 3])
                bottleneck_out = self.linear_middle(h_)

        out = self.blocks[-1](h)

        out = out.view(out.size(0), 1, self.resolution, self.resolution)

        # 返回 residual_features 中的某个四维特征图
        return residual_features, h, bottleneck_out, out


def D_unet_arch(ch=64, attention='64'):
    arch = {}

    arch[256] = {'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 8, 16, 8 * 2, 8 * 2, 4 * 2, 2 * 2, 1 * 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 8, 8, 4, 2, 1, 1]],
                 'downsample': [True] * 6 + [False] * 6,
                 'upsample': [False] * 6 + [True] * 6,
                 'resolution': [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 13)}}

    return arch


# Conv block for the discriminator encoder
class EBlock(nn.Module):
    def __init__(self, inc, outc, use_SN=True, activation='relu'):
        super(EBlock, self).__init__()
        self.inc, self.outc = inc, outc

        self.use_SN = use_SN
        self.activation = get_non_linearity(activation)()

        # Conv layer
        self.conv = nn.Conv2d(self.inc, self.outc, 4, stride=2, padding=1)

        if self.use_SN:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.activation(self.conv(x))


# Conv block for the discriminator decoder
class DBlock(nn.Module):
    def __init__(self, inc, outc, use_SN=True, activation='relu'):
        super(DBlock, self).__init__()
        self.inc, self.outc = inc, outc

        self.use_SN = use_SN
        self.activation = get_non_linearity(activation)()

        # Conv layer
        self.conv = nn.Conv2d(self.inc, self.outc, 3, stride=1, padding=1)

        if self.use_SN:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.activation(self.conv(F.interpolate(x, scale_factor=2, mode='nearest')))


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=False)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=False)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
    def __init__(self, ch, use_SN=True, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.to_q = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.to_key = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.to_value = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.output = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        if use_SN:
            self.to_q, self.to_key, self.to_value, self.output = spectral_norm(self.to_q), spectral_norm(
                self.to_key), spectral_norm(self.to_value), spectral_norm(self.output)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        query = self.to_q(x)
        # downsample key and value's spatial size
        key = F.max_pool2d(self.to_key(x), [2, 2])
        value = F.max_pool2d(self.to_value(x), [2, 2])
        # Perform reshapes
        query = query.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        key = key.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        value = value.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        attn_map = F.softmax(torch.bmm(query.transpose(1, 2), key), -1)
        # Attention map times g path
        o = self.output(torch.bmm(value, attn_map.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x
