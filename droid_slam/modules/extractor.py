import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.fusion import SelfAttFuser, DeformAttFuser
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


DIM=32

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False, args=None):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM,  stride=1)
        self.layer2 = self._make_layer(2*DIM, stride=2)
        self.layer3 = self._make_layer(4*DIM, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(4*DIM, output_dim, kernel_size=1)

        if self.multidim:
            self.layer4 = self._make_layer(256, stride=2)
            self.layer5 = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6 = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7 = self._make_layer(128, stride=1)

            self.up1 = nn.Conv2d(512, 256, 1)
            self.up2 = nn.Conv2d(256, 128, 1)
            self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)


class DualEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False, args=None):
        super(DualEncoder, self).__init__()
        self.encoder1 = BasicEncoder(output_dim, norm_fn, dropout, multidim)
        self.encoder2 = BasicEncoder(output_dim, norm_fn, dropout, multidim)
        if self.args.fusion_method == 'self_att':
            self.fuser1 = SelfAttFuser(DIM)
            self.fuser2 = SelfAttFuser(2*DIM)
            self.fuser3 = SelfAttFuser(4*DIM)
        elif self.args.fusion_method == 'deform_att':
            self.fuser1 = DeformAttFuser(DIM)
            self.fuser2 = DeformAttFuser(2*DIM)
            self.fuser3 = DeformAttFuser(4*DIM)
        elif self.args.fusion_method == 'concat':
            pass
        else:
            raise ValueError('Invalid fusion method')

    def forward(self, x):
        if self.args.fusion_method == 'concat':
            return self.forwardConcat(x)
        elif self.args.fusion_method == 'self_att' or self.args.fusion_method == 'deform_att':
            return self.forwardFusion(x)
        
    def forwardConcat(self, x):
        b, n, c1, h1, w1 = x.shape
        x1 = self.encoder1.forward(x[:,0:1])
        x2 = self.encoder2.forward(x[:,1:2])
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)
        
        
        
    
    def forwardFusion(self, x):
        b, n, c1, h1, w1 = x.shape
        x1 = x[:, :n//2]
        x2 = x[:, n//2:]

        x1 = self.encoder1.conv1(x1)
        x1 = self.encoder1.norm1(x1)
        x1 = self.encoder1.relu1(x1)
        
        x2 = self.encoder2.conv1(x2)
        x2 = self.encoder2.norm1(x2)
        x2 = self.encoder2.relu1(x2)

        x1 = self.encoder1.layer1(x1)
        x2 = self.encoder2.layer1(x2)
        x1, x2 = self.fuser1(x1, x2)
       
        x1 = self.encoder1.layer2(x1)
        x2 = self.encoder2.layer2(x2)
        x1, x2 = self.fuser2(x1, x2)
        
        x1 = self.encoder1.layer3(x1)
        x2 = self.encoder2.layer3(x2)
        x1, x2 = self.fuser3(x1, x2)

        x = torch.stack([x1, x2], dim=1)
        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)
        



