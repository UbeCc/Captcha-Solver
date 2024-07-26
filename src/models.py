import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10, captcha_length=5):
        super(CNN, self).__init__()
        self.captcha_length = captcha_length
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)        
        self.fc1 = nn.Linear(128 * 7 * 20, 512)
        self.fc2 = nn.Linear(512, num_classes * captcha_length)  # 修改为输出 num_classes * captcha_length
        self.dropout = nn.Dropout(0.3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 7 * 20)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)        
        x = x.view(-1, self.captcha_length, 10)
        return x