import torch
import torch as t
from torch.nn import functional as F
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm3d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv3d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm3d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv3d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm3d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv3d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))  # output size = (1, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        return x_4

class ACMBlock(nn.Module):
    def __init__(self, in_channels):
        super(ACMBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.k_conv = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels, (1, 1,1), groups=16),
        )

        self.q_conv = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels, (1, 1,1), groups=16),
        )

        self.global_pooling = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels // 2, (1, 1,1)),
            nn.ReLU(),
            nn.Conv3d(self.out_channels // 2, self.out_channels, (1, 1,1)),
            nn.Sigmoid()
        )


        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.normalize = nn.Softmax(dim=3)

    def _get_normalized_features(self, x):
        ''' Get mean vector by channel axis '''
        c_mean = self.avgpool(x)
        return c_mean

    def _get_orth_loss(self, K, Q):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        orth_loss = cos(K, Q)
        orth_loss = t.mean(orth_loss, dim=0)
        return orth_loss

    def _get_orth_loss_ACM(self, K, Q, c):
        orth_loss = t.mean(K * Q / c, dim=1, keepdim=True)
        return orth_loss

    def forward(self, x1, x2):

        mean_x1 = self._get_normalized_features(x1)
        mean_x2 = self._get_normalized_features(x2)
        x1_mu = x1 - mean_x1
        x2_mu = x2 - mean_x2

        K = self.k_conv(x1_mu)
        Q = self.q_conv(x2_mu)

        b, c, h, w,l = K.shape

        K = K.view(b, c, 1, h * w*l)
        K = self.normalize(K)
        K = K.view(b, c, h, w,l)


        Q = Q.view(b, c, 1, h * w*l)
        Q = self.normalize(Q)
        Q = Q.view(b, c, h, w,l)

        K = t.einsum('nchwl,nchwl->nc', [K, x1_mu])
        Q = t.einsum('nchwl,nchwl->nc', [Q, x2_mu])
        K = K.view(K.shape[0], K.shape[1], 1, 1,1)
        Q = Q.view(Q.shape[0], Q.shape[1], 1, 1,1)

        channel_weights1 = self.global_pooling(mean_x1)
        channel_weights2 = self.global_pooling(mean_x2)

        out1 = x1 + K - Q
        out2 = x2 + K - Q

        out1 = channel_weights1 * out1
        out2 = channel_weights2 * out2

        orth_loss = self._get_orth_loss(K, Q)

        return out1, out2, orth_loss

class mismatch_resnet(nn.Module):
    def __init__(self):
        super(mismatch_resnet, self).__init__()

        self.encoder =resnet18(1)

        self.acm=ACMBlock(in_channels=512)
        self.avgpool_1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool_2 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_IDH = nn.Linear(1024, 256)
        self.fc_1p19q = nn.Linear(1024, 256)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initializes weights using He et al. (2015)."""
        if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)
            # nn.init.constant_(m.bias.UCI, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            # nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1,x2):

        x_acm_1 = self.encoder(x1)
        x_acm_2 = self.encoder(x2)

        x_cmtb_1, x2_cmtb_2, orth_loss = self.acm(x_acm_1, x_acm_2)

        x_cmtb_1 = self.avgpool_1(x_cmtb_1)
        x_1 = torch.flatten(x_cmtb_1, 1)

        x2_cmtb_2 = self.avgpool_2(x2_cmtb_2)
        x_2 = torch.flatten(x2_cmtb_2, 1)

        out = torch.cat([x_1, x_2], 1)
        x_IDH = self.fc_IDH(out)
        x_IDH = F.leaky_relu(x_IDH)
        # 7. Linear + Classifier
        x_1p19q = self.fc_1p19q(out)
        x_1p19q = F.leaky_relu(x_1p19q)

        return  x_IDH,x_1p19q,orth_loss,out


def resnet18(num_classes=1, include_top=True):

    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)




