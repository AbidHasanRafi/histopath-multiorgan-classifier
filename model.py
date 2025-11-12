import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class MicroAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.conv(x)
        return x * attention_weights

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.gelu(x)

class EfficientBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.attention = MicroAttention(out_channels)
        
        self.use_residual = (in_channels == out_channels and stride == 1)
        if self.use_residual:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.attention(x)
        return x + residual if self.use_residual else x

class UltraLightHistoNet(nn.Module):
    def __init__(self, num_classes=14, base_channels=16):
        super().__init__()
        self.num_classes = num_classes
        
        # Store intermediate features for XAI
        self.features = {}
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        
        self.layer1 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer2 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, base_channels * 16, 2, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 16, base_channels * 8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 8, num_classes)
        )
        
        # Register hooks for feature extraction
        self._register_hooks()
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(EfficientBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(EfficientBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _register_hooks(self):
        def get_feature_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hooks for different layers for XAI
        self.stem.register_forward_hook(get_feature_hook('stem'))
        self.layer1.register_forward_hook(get_feature_hook('layer1'))
        self.layer2.register_forward_hook(get_feature_hook('layer2'))
        self.layer3.register_forward_hook(get_feature_hook('layer3'))
        self.layer4.register_forward_hook(get_feature_hook('layer4'))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        self.features.clear()
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
