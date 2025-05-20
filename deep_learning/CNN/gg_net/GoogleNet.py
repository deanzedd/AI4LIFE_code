import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

#device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f'Device: {device}')

#parameter
batch_size = 64
learning_rate = 0.01
num_epochs = 50

#0) prepare data
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

Train_val_datasets = torchvision.datasets.CIFAR10(root='C:/Users/THE ANH/Desktop/code/AI/data', train=True,
                                        download=True, transform=transform_train)
Test_datasets = torchvision.datasets.CIFAR10(root='C:/Users/THE ANH/Desktop/code/AI/data', train = False,
                                        download=True, transform=transform_train)
#split datasets into data and valid set
train_size = int(0.9*len(Train_val_datasets))
val_size = len(Train_val_datasets) - train_size
Train_datasets, val_datasets = random_split(Train_val_datasets, [train_size, val_size])

Train_loader = torch.utils.data.DataLoader(Train_datasets, batch_size = batch_size, shuffle = True)
Val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
Test_loader = torch.utils.data.DataLoader(Test_datasets, batch_size = batch_size, shuffle = False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#model
class Inception(nn.Module):
  def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
    super(Inception, self).__init__(**kwargs)

    #path 1 is 1x1 convolutional layer
    self.p1_1 = nn.Conv2d(in_channels=in_channels,out_channels=c1, kernel_size=1,padding='same',bias=False)
    self.p1_2 = nn.BatchNorm2d(c1)

    #path 2 is 1x1 conv layer followed by 3x3 conv layer
    self.p2_1 = nn.Conv2d(in_channels=in_channels,out_channels=c2[0], kernel_size=1,padding='same',bias=False)
    self.p2_2 = nn.BatchNorm2d(c2[0])
    self.p2_3 = nn.Conv2d(in_channels=c2[0],out_channels=c2[1], kernel_size=3, padding=1,bias=False)
    self.p2_4 = nn.BatchNorm2d(c2[1])

    #path 3 is 1x1 conv layer followed by 5x5 conv layer
    self.p3_1 = nn.Conv2d(in_channels=in_channels,out_channels=c3[0], kernel_size=1,padding='same',bias=False)
    self.p3_2 = nn.BatchNorm2d(c3[0])
    self.p3_3 = nn.Conv2d(in_channels=c3[0],out_channels=c3[1], kernel_size=3,padding='same',bias=False)
    self.p3_4 = nn.BatchNorm2d(c3[1])

    #path 4 is 3x3 maxPool followed by 1x1 conv layer
    self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    self.p4_2 = nn.Conv2d(in_channels=in_channels,out_channels=c4, kernel_size=1,padding='same',bias=False)
    self.p4_3 = nn.BatchNorm2d(c4)

  def forward(self, x):
    # Path 1: Conv1x1 -> BatchNorm -> ReLU
    p1 = F.relu(self.p1_2(self.p1_1(x)))

    # Path 2: Conv1x1 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU
    p2 = F.relu(self.p2_4(self.p2_3(F.relu(self.p2_2(self.p2_1(x))))))

    # Path 3: Conv1x1 -> BatchNorm -> ReLU -> Conv5x5 -> BatchNorm -> ReLU
    p3 = F.relu(self.p3_4(self.p3_3(F.relu(self.p3_2(self.p3_1(x))))))

    # Path 4: MaxPool3x3 (stride 1) -> Conv1x1 -> BatchNorm -> ReLU
    p4 = F.relu(self.p4_3(self.p4_2(self.p4_1(x))))

    # Concatenate the outputs on the channel dimension
    return torch.cat((p1, p2, p3, p4), dim=1)

class GoogleNet(nn.Module):
  def __init__(self):
     super(GoogleNet, self).__init__()
     self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
     self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU())

     self.layer2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=64,out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(192),
                                 nn.ReLU())

     self.layer3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                 Inception(256, 128, (128, 192), (32, 96), 64))

     self.layer4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                 Inception(512, 160, (112, 224), (24, 64), 64),
                                 Inception(512, 128, (128, 256), (24, 64), 64),
                                 Inception(512, 112, (144, 288), (32, 64), 64),
                                 Inception(528, 256, (160, 320), (32, 128), 128))
     self.layer5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                 Inception(832, 384, (192, 384), (48, 128), 128))

     self.last_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Dropout(0.4),
                                     nn.Flatten(),
                                     nn.Linear(1024, 10))

  def forward(self, x):
    x = self.layer1(x)
    x = self.pool(x)
    x = self.layer2(x)
    x = self.pool(x)
    x = self.layer3(x)
    x = self.pool(x)
    x = self.layer4(x)
    x = self.pool(x)
    x = self.layer5(x)
    x = self.last_layer(x)
    x = x.view(x.size(0), -1)
    return x

path='googlenet_cifar10_epoch7.pth'
#1) model
model = GoogleNet().to(device)
#model.load_state_dict(torch.load(path))
#model.eval()
print('đã tại lại model')

#2) loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# khởi tạo các list để lưu trữ accuracy
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

#3) training loop
n_total_steps = len(Train_loader)
for epoch in tqdm(range(num_epochs)):
    #training
    model.train()
    train_correct =0
    train_total =0
    train_loss_epoch = 0.0
    for i, (images, labels) in enumerate(Train_loader):
        images = images.to(device)
        labels = labels.to(device)
    
        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #accuracy ở batch
        train_loss_epoch += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    #accuray qua mỗi epoch    
    train_accuracy = 100 * train_correct / train_total  
    train_loss_epoch /= train_total
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss_epoch)
    
    #test on validation test
    model.eval()
    val_correct = 0
    val_total=0
    val_loss_epoch = 0.0
    with torch.no_grad():
        for images_val, labels_val in Val_loader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, labels_val)

            val_loss_epoch += loss_val.item() * images_val.size(0)
            _, predicted_val = torch.max(outputs_val.data, 1)
            val_total += labels_val.size(0)
            val_correct += (predicted_val == labels_val).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    val_loss_epoch /= val_total
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss_epoch)
      
    torch.save(model.state_dict(), f'googlenet_cifar10_epoch{epoch+1}.pth') # Provide the model's state_dict and a file path
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_epoch:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss_epoch:.4f}, Val Accuracy: {val_accuracy:.2f}%')
  
#4) test 
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in Test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        #value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        
    acc = 100* n_correct/n_samples
    print(f'accuracy = {acc}')
    print(n_samples)
    
    
# Vẽ biểu đồ accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png') # Lưu biểu đồ thành file (tùy chọn)
plt.show()

# Vẽ biểu đồ loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png') # Lưu biểu đồ thành file (tùy chọn)
plt.show()