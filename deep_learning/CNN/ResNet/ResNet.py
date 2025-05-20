import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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

# 0) prepare data
Train_val_datasets = torchvision.datasets.CIFAR10(root='C:/Users/THE ANH/Desktop/code/AI/data', train=True,
                                        download=True, transform=transform_train)
Test_datasets = torchvision.datasets.CIFAR10(root='C:/Users/THE ANH/Desktop/code/AI/data', train = False,
                                        download=True, transform=transform_train)
Train_size = int(0.9*len(Train_val_datasets))
val_size = len(Train_val_datasets) - Train_size
Train_datasets, val_datasets = random_split(Train_val_datasets, [Train_size, val_size])

Train_loader = torch.utils.data.DataLoader(Train_datasets, batch_size = batch_size, shuffle = True)
Val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
Test_loader = torch.utils.data.DataLoader(Test_datasets, batch_size = batch_size, shuffle = False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ResidualBlock(nn.Module):
    '''The Residual block of ResNet models.'''
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1, stride=strides)

        self.conv2 = nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1)
        #use 1x1 conv to make input and output have a same shape
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.functional.relu(Y)

class ResNet(nn.Module):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(16, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU(),
        )

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i==0 and not first_block:#nếu ko phải khối Residual Block đầu tiên x2 channels và giảm 2 lần height, width
                blk.append(ResidualBlock(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(ResidualBlock(num_channels))
        return nn.Sequential(*blk)

    def __init__(self, arch, lr=0.1, num_classes=10):
        super(ResNet, self).__init__()
        
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)
        ))
    
    def forward(self, x):
        for block in self.net:
            x = block(x)
        return x
    
arch_110 = [(18, 16), (18, 32), (18, 64)]

model = ResNet(arch=arch_110, lr=learning_rate).to(device)
print('model oke')

# 2) loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# khởi tạo các list để lưu trữ accuracy
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

#state của model được lưu tại file sau 
save_folder = 'C:/Users/THE ANH/Desktop/code/AI/deep_learning/ResNet/hist'

# 3) training loop
n_total_train = len(Train_loader)
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

    file_name = f'ResNet_cifar10_epoch{epoch+1}.pth'
    full_path = os.path.join(save_folder, file_name)
    torch.save(model.state_dict(), full_path) # Provide the model's state_dict and a file path
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




