import torch
import torch.nn as nn
from Structure.pytorch_emg_incept_cbam_model_4 import CustomCNN
# from pytorch_emg_incept_cbam_model5 import check_accuracy
import pickle
from torchmetrics import Precision,Recall,F1Score
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision import models
from torchsummary import summary

from image_loader import *
criterion=nn.CrossEntropyLoss()
test_accuracy_dict={}
test_loss_dict={}
train_loss_epoch_dict={}
loss_dict={}
train_accuracy_dict={}
precision_dict={}
recall_dict={}
F1_Score_dict={}
device="cuda"
num_classes=17

def check_accuracy(loader,model):
    print("...................")
    test_loss = []
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader :
            x=x.to(device)
            y=y.to(device)
            # x=x.reshape(x.shape[0],-1)
            scores=model(x)
            t_loss=criterion(scores, y)
            # test_loss[epochs]=t_loss

            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)


        # confusion Matrix


        #######################3
        print(f"||||||||||||||Got {num_correct}/{num_samples}  with accuracy {num_correct/num_samples*100:.2f}")
        ac=num_correct/num_samples*100

        # confusion metrics
        metric = MulticlassConfusionMatrix(num_classes=17).to(device)
        metric(predictions, y)
        fig_, ax_ = metric.plot()
        fig_.savefig(f'confusion_matrix{ac}.png')  # Save as PNG image

        # Precision
        precision = Precision(task='MULTICLASS',average='macro',num_classes=num_classes).to(device)  # For overall precision across classes
        precision(predictions, y)  # Update the metric
        overall_precision = precision.compute()

        # Recall
        recall = Recall(task='MULTICLASS', average='macro',num_classes=num_classes).to(device)
        recall(predictions,y)
        overall_recall=recall.compute()

        # F1 score
        F1=F1Score(task='MULTICLASS', average='macro',num_classes=num_classes).to(device)
        F1(predictions,y)
        overall_F1=F1.compute()

        print("Precision",overall_precision)
        print("Recall",overall_recall)
        print("F1",overall_F1)


        # model.train()
        return f"{num_correct/num_samples*100:.2f}", f"{t_loss:.4f}",overall_precision.item(),overall_recall.item(),overall_F1.item()



model = CustomCNN().to(device="cuda")
model_path="C:\\Users\\shitosu\\Desktop\\Training result\\Results\\model_4_incep_cbam_conv\\Model_78.74.pth"
x=torch.randn(2,3,224,224).to(device="cuda")
model.eval()
summary(model,(3,224,224))

# check_accuracy(test_loader,model)
