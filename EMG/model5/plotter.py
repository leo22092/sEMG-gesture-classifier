import matplotlib.pyplot
import os
import pickle
import matplotlib.pyplot as plt

# file_name="loss_data_model_incep_cbam_cnn.pkl"
file_name= "Training_results_model____.pkl"
if os.path.exists(file_name):
    print("Exists")
    with open(file_name,'rb') as f:
        metrics=pickle.load(f)
        train_acc=list(map(float,list(metrics["train_acc"].values())))
        val_acc=list(map(float,list(metrics["val_acc"].values())))
        train_loss=list(map(float,list(metrics["train_loss"].values())))
        val_loss=list(map(float,list(metrics["val_loss"].values())))
        train_loss_batchwise = list(map(float, list(metrics["train_loss_batchwise"].values())))
        precision=list(map(float,list(metrics["val_loss"].values())))
        recall=list(map(float,list(metrics["val_loss"].values())))
        F1_score=list(map(float,list(metrics["F1Score"].values())))
        batches=[i+1 for i in range(len(metrics["train_loss_batchwise"]))]
        epochs = [i+1 for i in range(len(metrics["F1Score"]))]

        print(train_acc)
        print(epochs)
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

plt.plot(epochs, train_loss, label='Precision')
plt.plot(epochs, val_loss, label='Recall')
plt.xlabel('Epochs')
plt.ylabel('P-R')
plt.legend()
plt.title('Precision Recall')
plt.savefig("Precision Recall.png")
plt.show()

plt.plot(epochs, train_loss, label='F1 Score')
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


# print(losses.keys())
# j=list(map(float,list(losses.values())))
#
# # plt.plot(int(losses.values()))
# # plt.plot([1,2,3,4,5,6,7],[1,4,9,16,25,36,49])
# plt.plot(j)
# plt.show()

