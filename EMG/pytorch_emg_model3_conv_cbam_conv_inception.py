import pickle
import torch.nn.functional as F

import torch.optim as optim
from pytorch_cbam import*
from inceeption_blocks import *
# device= ('cuda' if torch.cuda.is_available() else 'cpu')
device="cuda"
from image_loader import *

class CNN(nn.Module):
    def __init__(self,in_channel=3,num_classes=17):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))#same padding

        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))#haalfs the size
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))#same padding
        print(self.conv2.parameters())
        self.conv3=Conv_Block(in_channels=16,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv4=Conv_Block(64,192,kernel_size=3,stride=1,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inception=Inception_block(192,192,96,208,16,48, 64)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)



        self.fc1=nn.Linear(512,num_classes)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        # print("This is x,shape",x.shape)
        module = cbam(8).to(device)

        # print("cbam.shape",module(x).shape)
        x=module(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.maxpool1(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.maxpool2(x)
        x=self.inception(x)
        x=self.avgpool(x)
        print(x.shape)


        # print("Shape after relu",x.shape)
        print("forward_1")

        x=x.reshape(x.shape[0],-1)
        # print(x.shape)
        x=self.fc1(x)
        # print("forward")
        return x

model=CNN().to(device)
# x=torch.randn(64,1,28,28)
# print(model(x).shape)
# exit()
in_channel=3
num_classes=17
learning_rate=0.001
batch_size=128
num_epochs=100

# Data loader
# train_dataset=datasets.MNIST(root='pytorch__cnn\datasets',train=True,transform=transforms.ToTensor(),download=True)
# train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
#
# test_dataset=datasets.MNIST(root='pytorch__cnn\datasets',train=False,transform=transforms.ToTensor(),download=True)
# test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
# OVERFITTING
data,targets=next(iter(train_loader))

# Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
accuracy_dict={}
def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader :
            x=x.to(device)
            y=y.to(device)
            # x=x.reshape(x.shape[0],-1)
            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f"||||||||||||||Got {num_correct}/{num_samples}  with accuracy {num_correct/num_samples*100:.2f}")
        return f"{num_correct/num_samples*100:.2f}"
loss_dict={}
count_batch=0
for epochs in range(num_epochs):
    for batch_idx,(data,targets) in enumerate (train_loader):
        if epochs>=3 :
            learning_rate=0.0001
            for param_group in optimizer.param_groups:
                param_group['lr']=learning_rate
        # if epochs>=10 :
        #     learning_rate=0.00001
        #     for param_group in optimizer.param_groups:
        #         param_group['lr']=learning_rate


        data=data.to(device)
        targets=targets.to(device)

        # Forward
        scores=model(data)
        loss=criterion(scores,targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # print(f"Epoch:{epochs} Batch:{batch_idx} Loss: {loss}")
        count_batch += 1

        loss_dict[count_batch]=loss

        # Gradient descent or adam step
        optimizer.step()
        print(f"EPOCH {epochs} ,BATCH {batch_idx} ,Loss : {loss} Learning rate is lr,{[print(param_group['lr']) for param_group in optimizer.param_groups]}")
        # print(epochs,"....",float(loss))

        for param_group in optimizer.param_groups:
            print("Leaarning rate is ",param_group['lr'])
    accuracy=check_accuracy(test_loader, model)
    accuracy_dict[epochs]=accuracy
    print(accuracy)
    if float(accuracy)>70:
        torch.save(model.state_dict(), f"Model_{accuracy}.pth")
        torch.save(model,f"Model_{accuracy}.pkl")
        torch.onnx.export(model,x,"CustomNet.onnx")

# check_accuracy(test_loader,model)
def csv_writer(dictt,name):
    with open (f'{name}.pkl','wb') as f:
       pickle.dump(dictt,f)

csv_writer(loss_dict,"loss_data_model_incep_cbam_cnn")
csv_writer(accuracy_dict,"accuracy_data_model_incep_cbam_cnn")

