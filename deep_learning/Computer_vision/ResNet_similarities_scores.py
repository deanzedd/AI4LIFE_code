import numpy as np
import torch, torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader, random_split
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
# tạo 2 datasets: 1: 2 classes, 2: 3 classes
#load lại model lấy tham số sau khi train
#tính similarities score của model với 2 datasets

#device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f'device: {device}')

#parameter
batch_size = 64
learning_rate = 0.01
num_epochs = 50

# 0) prepare data
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    #tạm tắt để visualize
])

Train_val_datasets = torchvision.datasets.CIFAR10(root='C:/Users/THE ANH/Desktop/code/AI/data',
                                                  train=True, download=True, transform=transform_train)
Test_datasets = torchvision.datasets.CIFAR10(root='C:/Users/THE ANH/Desktop/code/AI/data',
                                             train=False, download=True, transform=transform_train)
#tắt transform của data

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes1 = ('cat', 'ship')
classes2 = ('cat', 'dog', 'ship')
'''
Train cho model 1
chứa 2 classes cat, ship

'''

class_to_index = Train_val_datasets.class_to_idx
#lấy chỉ số ở lớp mục tiêu
target_indices = [class_to_index[clc] for clc in classes1]
train_val_indices = [i for i, label in enumerate(Train_val_datasets.targets) if label in target_indices]
Train_val_cat_ship_datasets = Subset(Train_val_datasets, train_val_indices)

#lấy test data cho classes1
test_indices = [i for i, label in enumerate(Test_datasets.targets) if label in target_indices]
Test_cat_ship_datasets = Subset(Test_datasets, test_indices)

train_size1 = int(0.9*len(Train_val_cat_ship_datasets))
val_size1 = len(Train_val_cat_ship_datasets)-train_size1
Train_cat_ship_datasets, val_cat_ship_datasets = random_split(Train_val_cat_ship_datasets, [train_size1, val_size1])

Train_loader1 = DataLoader(Train_cat_ship_datasets, batch_size=batch_size, shuffle=True)
Val_loader1 = DataLoader(val_cat_ship_datasets, batch_size=batch_size, shuffle=False)
Test_loader1 = DataLoader(Test_cat_ship_datasets, batch_size=batch_size, shuffle=False)

'''
Train cho model 2
chứa 3 class cat, dog, ship
'''
# lấy data

#lấy chỉ số ở lớp mục tiêu
target_indices2 = [class_to_index[clc] for clc in classes2]
train_val_indices2 = [i for i, label in enumerate(Train_val_datasets.targets) if label in target_indices2]
Train_val_cat_ship_dog_datasets = Subset(Train_val_datasets, train_val_indices2)

#lấy test data cho classes2
test_indices2 = [i for i, label in enumerate(Test_datasets.targets) if label in target_indices2]
Test_cat_ship_dog_datasets = Subset(Test_datasets, test_indices2)

train_size2 = int(0.9*len(Train_val_cat_ship_dog_datasets))
val_size2 = len(Train_val_cat_ship_dog_datasets)-train_size2
Train_cat_ship_dog_datasets, val_cat_ship_dog_datasets = random_split(Train_val_cat_ship_dog_datasets, [train_size2, val_size2])

Train_loader2 = DataLoader(Train_cat_ship_dog_datasets, batch_size=batch_size, shuffle=True)
Val_loader2 = DataLoader(val_cat_ship_dog_datasets, batch_size=batch_size, shuffle=False)
Test_loader2 = DataLoader(Test_cat_ship_dog_datasets, batch_size=batch_size, shuffle=False)

#model ResNet

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
        #net chạy hết các layer
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)
        ))
        
        '''
        #net1 chạy đến lớp CNN cuối cùng
        self.net1 = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net1.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        '''
    
    def forward(self, x):
        for block in self.net:
            x = block(x)
        return x
    
    def forward_to_last_layer(self, x):
        for name, module in self.net.named_children():
            if name == 'last':
                break  # Dừng lại khi gặp module 'last'
            x = module(x)
        return x
    
        
arch_110 = [(18, 16), (18, 32), (18, 64)]
model1 = ResNet(arch=arch_110, lr=learning_rate).to(device=device)
model2 = ResNet(arch=arch_110, lr=learning_rate).to(device=device)
path1 = 'C:/Users/THE ANH/Desktop/code/AI/deep_learning/Computer_vision/hist/hist_ResNet/model1/ResNet_cifar10_cat_ship_epoch50.pth'
path2 = 'C:/Users/THE ANH/Desktop/code/AI/deep_learning/Computer_vision/hist/hist_ResNet/model2/ResNet_cifar10_cat_ship_dog_epoch50.pth'

model1.load_state_dict(torch.load(path1))
model1.eval()

model2.load_state_dict(torch.load(path2))
model2.eval()

#lấy ảnh 'cat' từ datasets
cat_image_sample = None
cat_label_idx = Test_datasets.class_to_idx['cat']

# tìm ảnh mèo xh dầu tiên trong data mình có
for img, label in Test_datasets:
    if label == cat_label_idx:
        cat_image_sample = img.unsqueeze(0) # Thêm batch dimension (batch size = 1)
        cat_image_sample = cat_image_sample.to(device)

#tính đầu ra của ảnh 'cat' qua layer CNN cuối cùng
embedding_model1 = model1.forward_to_last_layer(cat_image_sample)
embedding_model2 = model2.forward_to_last_layer(cat_image_sample)


# 5. Tính toán Độ tương đồng (Cosine Similarity)
# Sklearn's cosine_similarity yêu cầu input là mảng numpy 2D (số_mẫu, số_đặc_trưng)
# Chuyển tensor embedding về CPU và numpy (do numpy() yêu cầu tính toán trên CPU chứ ko trực tiếp trên GPU được), sau đó reshape
# .detach() tạo ra một bản sao của tensor gốc( do torch tạo ra computition graph theo dõi tính toán autograd), mà mình muốn tính toán
# mà ko ảnh hướng đến tensor gốc nên cần .detach()
embedding_model1_np = embedding_model1.cpu().detach().numpy().reshape(1, -1)
embedding_model2_np = embedding_model2.cpu().detach().numpy().reshape(1, -1)

# Tính Cosine Similarity
similarity_score = cosine_similarity(embedding_model1_np, embedding_model2_np)[0][0]
print(f'similarity_score: {similarity_score}')
if similarity_score > 0.7: # Ngưỡng tùy ý cho độ tương đồng cao
    print("- Điểm tương đồng cao có thể ngụ ý rằng:")
    print("  - Cả hai mô hình học được các đặc trưng rất tương tự cho ảnh 'cat' ở lớp này.")
    print("  - Việc Model 2 được huấn luyện thêm trên lớp 'dog' có thể không làm thay đổi đáng kể")
    print("    cách nó biểu diễn đặc trưng của 'cat' so với Model 1 (chỉ học cat, ship).")
elif similarity_score < 0.3: # Ngưỡng tùy ý cho độ tương đồng thấp
    print("- Điểm tương đồng thấp có thể ngụ ý rằng:")
    print("  - Việc huấn luyện trên tập dữ liệu khác nhau (đặc biệt là sự có mặt của lớp 'dog' trong Model 2)")
    print("    đã dẫn đến việc Model 2 học được một không gian đặc trưng hoặc các đặc trưng cho 'cat' khá khác biệt")
    print("    so với Model 1.")
    print("  - Điều này có thể phản ánh rằng 'dog' là một lớp gây nhiễu hoặc làm thay đổi cách phân tách ranh giới")
    print("    giữa các lớp mà Model 2 học được, ảnh hưởng đến biểu diễn của 'cat'.")
else:
    print("- Điểm tương đồng trung bình cho thấy có sự khác biệt vừa phải trong cách hai model biểu diễn ảnh 'cat'.")
    print("  Sự khác biệt này có thể đến từ sự ngẫu nhiên trong huấn luyện, hoặc ảnh hưởng nhẹ của việc thêm lớp 'dog'.")

