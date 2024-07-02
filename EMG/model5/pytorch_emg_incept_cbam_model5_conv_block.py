import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from pytorch_cbam import *
# device= ('cuda' if torch.cuda.is_available() else 'cpu')
device="cuda"
from inceeption_blocks import *
from matplotlib import pyplot as plt
from torchmetrics import Precision,Recall,F1Score
from torchmetrics.classification import MulticlassConfusionMatrix
from torchsummary import summary

class CustomCNN(nn.Module):
    def __init__(self,in_channels=3,num_classes=17):
        super(CustomCNN,self).__init__()

        self.conv1 = Conv_Block(in_channels=in_channels,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2=Conv_Block(64,192,kernel_size=3,stride=1,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception3a=Inception_block(192,64,96,128,16,32,32)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.conv3=nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1)
        self.module = cbam(64).to(device)
        self.conv4=Conv_Block(64,34,kernel_size=3,stride=1,padding=1)
        self.conv5 = Conv_Block(34, 17, kernel_size=3, stride=1, padding=1)




        self.fc1=nn.Linear(17*22*22,num_classes)

    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.inception3a(x)
        x=self.avgpool(x)
        x=self.conv3(x)
        # print("11111")
        # print(x.shape)
        x=self.module(x)
        # print("forward_1")
        x=self.conv4(x)
        # print(x.shape)
        x=self.conv5(x)
        x=x.reshape(x.shape[0],-1)
        # print(x.shape)
        x=self.fc1(x)
        # print("forward")
        return x
model = CustomCNN().to(device="cuda")
summary(model,(3,224,224))
