import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from image_loader import *
from cbam.pytorch_cbam import *
device= ('cuda' if torch.cuda.is_available() else 'cpu')

class SpectralImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SpectralImageClassifier, self).__init__()

        # Define input channels
        in_channels = 3  # Modify this based on your spectral image channels

        # Convolutional block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.25)

        # Convolutional block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=0.25)

        # Convolutional block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(p=0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)  # Adjust spatial dimensions based on input size
        self.drop4 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Convolutional block 1
        x = self.conv1(x)
        x = torch.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        # Convolutional block 2
        x = self.conv2(x)
        x = torch.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        # Convolutional block 3
        x = self.conv3(x)
        x = torch.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.drop3(x)

        # Flatten and fully connected layers
        x = x.reshape(-1, 128 * 7 * 7)  # Adjust spatial dimensions based on input size
        x = torch.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        print("Forward")
        return x


# Example usage
num_classes = 17
model = SpectralImageClassifier(num_classes).to(device)
in_channel=3
num_classes=17
learning_rate=0.01
batch_size=2
num_epochs=5
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

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
        print(f"Got {num_correct}/{num_samples}  with accuracy {num_correct/num_samples*100:.2f}")

# for epochs in range(num_epochs):
#     for batch_idx,(data,targets) in enumerate (train_loader):
#         data=data.to(device)
#         targets=targets.to(device)
#
#         # Forward
#         scores=model(data)
#         loss=criterion(scores,targets)
#
#         # Backward
#         optimizer.zero_grad()
#         loss.backward()
#         print(f"Epoch:{epochs} Batch:{batch_idx} Loss: {loss}")
#
#         # Gradient descent or adam step
#         optimizer.step()
#     print(f"FOr the EPOCH {epochs}")
#     check_accuracy(test_loader,model)
#     torch.save(model.state_dict(),f"model_epoch{epochs}.pth")
#
print(model(torch.randn(32,3,256,256).to(device)).shape)


check_accuracy(test_loader,model)
