import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from cbam.pytorch_cbam import *
# device= ('cuda' if torch.cuda.is_available() else 'cpu')
device="cuda"
from image_loader import *
from inceeption_blocks import *
from matplotlib import pyplot as plt
from torchmetrics import Precision,Recall,F1Score
from torchmetrics.classification import MulticlassConfusionMatrix


class CustomCNN(nn.Module):
    def __init__(self,in_channels=3,num_classes=8):
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




        self.fc1=nn.Linear(34*22*22,num_classes)

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

        x=x.reshape(x.shape[0],-1)
        # print(x.shape)
        x=self.fc1(x)
        # print("forward")
        return x

def csv_writer(dictt,name):
    with open (f'{name}.pkl','wb') as f:
       pickle.dump(dictt,f)

# def check_accuracy(loader,model):
#     print("...................")
#     test_loss = []
#     num_correct = 0
#     num_samples = 0
#     model.eval()
#
#     with torch.no_grad():
#         for x,y in loader :
#             x=x.to(device)
#             y=y.to(device)
#             # x=x.reshape(x.shape[0],-1)
#             scores=model(x)
#             t_loss=criterion(scores, y)
#             # test_loss[epochs]=t_loss
#
#             _,predictions=scores.max(1)
#             num_correct+=(predictions==y).sum()
#             num_samples+=predictions.size(0)
#
#
#         # confusion Matrix
#
#
#         #######################3
#         print(f"||||||||||||||Got {num_correct}/{num_samples}  with accuracy {num_correct/num_samples*100:.2f}")
#         ac=num_correct/num_samples*100
#
#         # confusion metrics
#         metric = MulticlassConfusionMatrix(num_classes=8).to(device)
#         metric(predictions, y)
#         print("y.shape",y.shape)
#         print("predictions shape",predictions.shape)
#         fig_, ax_ = metric.plot()
#         fig_.savefig(f'confusion_matrix{ac}.png')  # Save as PNG image
#
#         # Precision
#         precision = Precision(task='MULTICLASS',average='macro',num_classes=num_classes).to(device)  # For overall precision across classes
#         precision(predictions, y)  # Update the metric
#         overall_precision = precision.compute()
#
#         # Recall
#         recall = Recall(task='MULTICLASS', average='macro',num_classes=num_classes).to(device)
#         recall(predictions,y)
#         overall_recall=recall.compute()
#
#         # F1 score
#         F1=F1Score(task='MULTICLASS', average='macro',num_classes=num_classes).to(device)
#         F1(predictions,y)
#         overall_F1=F1.compute()
#
#         print("Precision",overall_precision)
#         print("Recall",overall_recall)
#         print("F1",overall_F1)
#
#
#         # model.train()
#         return f"{num_correct/num_samples*100:.2f}", f"{t_loss:.4f}",overall_precision.item(),overall_recall.item(),overall_F1.item()


def check_accuracy(loader, model):
    print("...................")
    test_loss = []
    num_correct = 0
    num_samples = 0
    model.eval()

    # Initialize TP, FP, FN, true_labels, pred_labels for F1 score and confusion matrix
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            t_loss = criterion(scores, y)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            # Update TP, FP, FN
            for i in range(num_classes):
                TP[i] += ((predictions == y) & (y == i)).sum().item()
                FP[i] += ((predictions != y) & (predictions == i)).sum().item()
                FN[i] += ((predictions != y) & (y == i)).sum().item()

            # Collect true labels and predictions for F1 score and confusion matrix
            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

    # Calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    test_accuracy=(num_correct/num_samples)*100
    # float(round(test_accuracy,2))

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = torch.mean(f1[torch.isfinite(f1)])  # Ignore NaN values

    # Calculate confusion matrix
    conf_matrix = torch.zeros(num_classes,num_classes)
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t, p] += 1

    print(f"Precision: {precision}")
    print(f"Recall: {recall}type= {type(recall)}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    return test_accuracy,t_loss,precision,recall,f1,conf_matrix
#
in_channel=3
num_classes=8
learning_rate=0.01
batch_size=128
num_epochs=3

model=CustomCNN().to(device)


# data,targets=next(iter(train_loader))
# Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
test_accuracy_dict={}
test_loss_dict={}
train_loss_epoch_dict={}
loss_dict={}
train_accuracy_dict={}
precision_dict={}
recall_dict={}
F1_Score_dict={}
conf_matrix_dict={}

if __name__ =="__main__":

    for epochs in range(num_epochs):
        total_correct = 0
        total_samples = 0
        count_batch=1
        for batch_idx,(data,targets) in enumerate (train_loader):



            data=data.to(device)
            targets=targets.to(device)

            # Forward
            scores=model(data)
            loss=criterion(scores,targets)
            _, predicted = torch.max(scores, 1)

            # Update the running total of correct predictions and samples
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # print(f"Epoch:{epochs} Batch:{batch_idx} Loss: {loss}")
            count_batch += 1

            loss_dict[count_batch]=loss

            if float(loss)<=1.5:
                learning_rate=0.0001
                for param_group in optimizer.param_groups:
                    param_group['lr']=learning_rate
            if float(loss)<0.6:
                learning_rate=0.00001
                for param_group in optimizer.param_groups:
                    param_group['lr']=learning_rate


            # Gradient descent or adam step
            optimizer.step()
            print(f"EPOCH {epochs} ,BATCH {batch_idx} ,Loss : {loss} Learning rate is lr,")
            # print(epochs,"....",float(loss))
        test_accuracy_dict[epochs],test_loss_dict[epochs],precision_dict[epochs],recall_dict[epochs],F1_Score_dict[epochs],conf_matrix_dict[epochs]= check_accuracy(test_loader, model)

        for param_group in optimizer.param_groups:
            print("Leaarning rate is ",param_group['lr'])

        train_accuracy = 100 * total_correct / total_samples
        train_accuracy_dict[epochs]=train_accuracy
        print(f'Epoch {epochs + 1}:Train Accuracy = {train_accuracy:.2f}%')
        print(f'Epoch {epochs + 1}:Test Accuracy = {test_accuracy_dict[epochs]:.2f}%')
        print(f'Epoch {epochs + 1}:Test loss = {test_loss_dict[epochs]}%')
        # print(f'Epoch {epochs + 1}:Train loss = {torch.max(torch.tensor(map(float,list(loss_dict.values()))))}%')
        print(f'Epoch {epochs + 1}:Train loss = {max(loss_dict.values()):.2f}%')
        train_loss_epoch_dict[epochs]=f'{max(loss_dict.values()):4f}'

    train_results={}
    train_results["train_acc"]=train_accuracy_dict
    train_results["val_acc"]=test_accuracy_dict
    train_results["val_loss"]=test_loss_dict
    train_results["train_loss_batchwise"]=loss_dict
    train_results["Precision"]=precision_dict
    train_results["Recall"]=precision_dict
    train_results["F1Score"]=F1_Score_dict

    csv_writer(train_results,".\
    Training_results_model____")
    x=torch.randn(32,3,224,224).to(device)

    # torch.onnx.export(model,x,"CustomNet.onnx")
    # model=CustomCNN().to(device)
    # x=model(x)
