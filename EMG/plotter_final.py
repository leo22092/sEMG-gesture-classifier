import os
import pickle
import matplotlib.pyplot as plt

# data.pkl contains the results data
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


# Train and validation accuracy VS epoch plot
# Note:Taking only smallest loss in each batch so first batches maximumum loss is not plotted
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#Adding Title
plt.title('Training and Validation Accuracy')
#saving figure with name "Training Vs Validation Accuracy.png" in current directory
plt.savefig(".\Training Vs Validation Accuracy.svg")
plt.show()



# Train and validation Loss VS epoch plot
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("Training and Validation Loss.svg")
plt.show()


# Precision Recall VS Epochs Plot
plt.plot(epochs, precision, label='Precision')
plt.plot(epochs, recall, label='Recall')
plt.xlabel('Epochs')
plt.ylabel('P-R')
plt.legend()
plt.title('Precision Recall')
plt.savefig("Precision Recall.svg")
plt.show()


# F1 score VS epoch plot
plt.plot(epochs, F1_score, label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 Score')
plt.savefig("F1 Score.svg")
plt.show()


# Train loss vs Batch Plot (used for adjusting learning rates)
plt.plot(batches, train_loss_batchwise, label='train_loss_batchwise')
plt.title("Train Loss Batchwise")
plt.savefig("Train Loss Batchwise.svg")
plt.show()
