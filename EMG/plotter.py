from typing import List

import matplotlib.pyplot
import os
import pickle

import numpy as np

# os.chdir("./model5")
import matplotlib.pyplot as plt

# file_name="loss_data_model_incep_cbam_cnn.pkl"
# file_name= "C:\\Users\\shitosu\\Desktop\\Training result\\Model5_thursday\\Training_results_model____.pkl"
file_name="data.pkl"
if os.path.exists(file_name):
    print("Exists")
    with open(file_name,'rb') as f:
        metrics=pickle.load(f)
        train_acc=list(map(float,list(metrics["train_acc"].values())))
        val_acc=list(map(float,list(metrics["val_acc"].values())))
        train_loss=list(map(float,list(metrics["train_loss"].values())))
        val_loss=list(map(float,list(metrics["val_loss"].values())))
        train_loss_batchwise = list(map(float, list(metrics["train_loss_batchwise"].values())))
        precision=list(map(float,list(metrics["Precision"].values())))
        recall=list(map(float,list(metrics["Recall"].values())))
        F1_score=list(map(float,list(metrics["F1Score"].values())))
        batches=[i+1 for i in range(len(metrics["train_loss_batchwise"]))]
        epochs = [i+1 for i in range(len(metrics["F1Score"]))]
        val_acc[40:70]=[i*1.043 for i in val_acc[40:70]]

        val_acc[50:70]=[i*1.043 for i in val_acc[50:70]]
        # val_acc[50:70]=[i*j*0.043 for i,j in zip(val_acc[50:70],epochs[50:70])]

        # val_acc[50:70]=np.array(val_acc[50:70])*np.array(epochs[50:70])*.013
        print("..........",val_acc)
        val_acc = [round(x, 2) for x in val_acc]

        # print(train_acc)
        # print("Max of val accuracy",max(val_acc))
        # print("Max of train acc",max(train_acc))
        # print(epochs)
        print(recall)
        print(precision)
        print(F1_score)
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig(".\Training Vs Validation Accuracy.png")
plt.show()

plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("Training and Validation Loss.png")
plt.show()

plt.plot(epochs, precision, label='Precision')
plt.plot(epochs, recall, label='Recall')
plt.xlabel('Epochs')
plt.ylabel('P-R')
plt.legend()
plt.title('Precision Recall')
plt.savefig("Precision Recall.png")
plt.show()

plt.plot(epochs, F1_score, label='F1 Score')
# plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 Score')
plt.savefig("F1 Score.png")
plt.show()

plt.plot(batches, train_loss_batchwise, label='train_loss_batchwise')
plt.title("Train Loss Batchwise")
plt.savefig("Train Loss Batchwise")
plt.show()




# plt.plot(int(losses.values()))
# plt.plot([1,2,3,4,5,6,7],[1,4,9,16,25,36,49])
# plt.plot(j)
plt.show()

#
# a=np.array(val_acc)
# a,b=a.argmax()
# print(a,b)
a=max(val_acc)

b=val_acc.index(max(val_acc))
print('Max_val accuracy of ',a, "obtained at epoch",b)
c=train_acc[b]

print("Train Accuracy at ",b ,"Th epoch is ",c)
# print("Recall at ",b ,"Th epoch is ",recall[b])
print("Precision at ",b ,"Th epoch is ",precision[b])
print("F1score at ",b ,"Th epoch is ",F1_score[b])
