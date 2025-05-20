# grandfather_directory/child_directory_B/file_b.py
'''
grandfather_directory/
├── child_directory_A/
│   └── file_a.py
└── child_directory_B/
    └── file_b.py  # File này muốn import từ file_a.py
'''
import sys
import os

# Lấy đường dẫn tuyệt đối của file hiện tại (file_b.py)
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Lấy đường dẫn đến thư mục cha (child_directory_B)
parent_dir = os.path.dirname(current_file_dir)

# Lấy đường dẫn đến thư mục "ông" (grandfather_directory)
grandfather_dir = os.path.dirname(parent_dir)

# Lấy đường dẫn đến thư mục "cụ" 
cu_dir = os.path.dirname(grandfather_dir)


# Thêm đường dẫn thư mục "ông" vào sys.path
# Kiểm tra để tránh thêm trùng lặp
if cu_dir not in sys.path:
    sys.path.append(cu_dir)
    print(f"Added {cu_dir} to sys.path") # Dòng debug

# Bây giờ bạn có thể import từ các thư mục con của "ông"
# Cú pháp import sẽ là từ thư mục con (child_directory_A)
'''
try:
    from child_directory_A.file_a import hello_from_a, my_variable
    print("Import successful!")

    hello_from_a()
    print(f"Value from file_a: {my_variable}")

except ImportError as e:
    print(f"Import failed: {e}")
    print("Please check your directory structure and file names.")
'''


 
'''
#Cách Hoạt động của GradCAM (Gradient-weighted Class Activation Mapping)
# là kỹ thuật thuộc nhóm XAI (Explained AI)
tạo ra bản đồ nhiệt(heatmap) làm nổi bật các vùng trong ảnh đầu vào quan trọng
nhất đối với dự đoán của mạng nơ-ron tích chập (CNN) cho một lớp cụ thể.  

#Cách hoạt động
1) Forward pass: đưa ảnh đầu vào đến gần lớp cuối CNN
2) Xác định lớp mục tiêu: chọn đối tượng mà mô hình muốn giải thích
(vd: mô hình phân loạilà con mèo thì tại sao nó lại là con mèo)
3) x: tính gradient của điểm số(logits hoặc xs) của lớp mục tiêu đối với
features maps của lớp tích chập chọn ở bước 1)
Gradient này cho biết mức độ nhạy cảm của điểm số lớp mục tiêu đối với sự thay đổi giá trị tại mỗi vị trí trong mỗi bản đồ đặc trưng.
4) Tính Trọng số cho Bản đồ Đặc trưng: Đối với mỗi bản đồ đặc trưng,
tính giá trị trung bình của tất cả các gradient trên không gian (chiều rộng và chiều cao) của bản đồ đó.
Giá trị trung bình này được coi là trọng số (weight) cho bản đồ đặc trưng đó,
biểu thị mức độ "quan trọng" tổng thể của bản đồ đặc trưng đó đối với lớp mục tiêu.
Một bản đồ đặc trưng có trọng số dương cao nghĩa là nó chứa các đặc trưng giúp tăng điểm số cho lớp mục tiêu.
5) Kết hợp Bản đồ Đặc trưng với Trọng số: Nhân mỗi bản đồ đặc trưng với trọng số tương ứng vừa tính được.
Điều này làm nổi bật các đặc trưng trong mỗi bản đồ có liên quan tích cực đến lớp mục tiêu.
6) Áp dụng Hàm ReLU: Áp dụng hàm kích hoạt ReLU (Rectified Linear Unit) cho kết quả kết hợp ở bước 5.
Bước này loại bỏ các giá trị âm. Chúng ta chỉ quan tâm đến các vùng trong ảnh mà các đặc trưng của
chúng tăng khả năng dự đoán lớp mục tiêu, không quan tâm đến những vùng làm giảm khả năng đó. Kết quả11:21:06 sau ReLU là bản đồ nhiệt thô (raw heatmap).
7) Thay đổi Kích thước (Upsampling): Thay đổi kích thước của bản đồ nhiệt thô để nó có cùng kích thước với ảnh đầu vào ban đầu.
8) Phủ lên Ảnh gốc: Phủ bản đồ nhiệt đã thay đổi kích thước lên trên ảnh gốc.
Các vùng có màu sắc "nóng" (thường là màu đỏ, vàng) trên bản đồ nhiệt sẽ cho thấy các pixel
hoặc vùng trong ảnh gốc có đóng góp lớn nhất vào dự đoán của mô hình cho lớp mục tiêu.


'''
'''
from CNN.gg_net.GoogleNet import Inception, GoogleNet
from CNN.ResNet.ResNet import ResNet
'''

import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from datasets import load_dataset
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import random_split, DataLoader, Subset
import torch.nn as nn
import torchvision
from typing import List, Callable, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torchvision.models import resnet101
from XAI_hugging_face import run_grad_cam_on_image, display

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
 

#1) download data
#2) chọn model và cho data chạy dọc theo các lớp
#3) visualize các features nổi bật quyết định đến đầu ra của model



'''
Task 1: xem model ResNet(CNN nói chung) học các features để phân biệt như nào
b1: model 1:ResNet train trên cat, ship datasets
b2: model 2:ResNet train trên cat, dog, ship datasets
b3: visualize tất cả ra để biết model học ntn
'''

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
    #transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
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


arch_110 = [(18, 16), (18, 32), (18, 64)]


#khởi tạo tham số mặc định cho model
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_value = 36
set_seed(seed_value)
model1 = ResNet(arch=arch_110, lr=learning_rate).to(device=device)

set_seed(seed_value)
model2 = ResNet(arch=arch_110, lr=learning_rate).to(device)


# 2) loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)


# khởi tạo các list để lưu trữ accuracy
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

#state của model được lưu tại file sau 
save_folder = 'C:/Users/THE ANH/Desktop/code/AI/deep_learning/Computer_vision/hist/hist_ResNet/model1'
save_folder2 = 'C:/Users/THE ANH/Desktop/code/AI/deep_learning/Computer_vision/hist/hist_ResNet/model2'

#Training và test model 1
'''
n_total_train = len(Train_loader1)
for epoch in tqdm(range(num_epochs)):
    #training
    model1.train()
    train_correct =0
    train_total =0
    train_loss_epoch = 0.0
    for i, (images, labels) in enumerate(Train_loader1):
        images = images.to(device)
        labels = labels.to(device)
    
        #forward pass
        outputs = model1(images)
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
    model1.eval()
    val_correct = 0
    val_total=0
    val_loss_epoch = 0.0
    with torch.no_grad():
        for images_val, labels_val in Val_loader1:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = model1(images_val)
            loss_val = criterion(outputs_val, labels_val)

            val_loss_epoch += loss_val.item() * images_val.size(0)
            _, predicted_val = torch.max(outputs_val.data, 1)
            val_total += labels_val.size(0)
            val_correct += (predicted_val == labels_val).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    val_loss_epoch /= val_total
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss_epoch)

    file_name = f'ResNet_cifar10_cat_ship_epoch{epoch+1}.pth'
    full_path = os.path.join(save_folder, file_name)
    torch.save(model1.state_dict(), full_path) # Provide the model's state_dict and a file path
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_epoch:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss_epoch:.4f}, Val Accuracy: {val_accuracy:.2f}%')

#4) test 
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in Test_loader1:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model1(images)
        
        #value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        
    acc = 100* n_correct/n_samples
    print(f'accuracy = {acc}')
    print(n_samples)
'''

#Training and test model 2
'''
# 3) training loop
n_total_train = len(Train_loader2)
for epoch in tqdm(range(num_epochs)):
    #training
    model2.train()
    train_correct =0
    train_total =0
    train_loss_epoch = 0.0
    for i, (images, labels) in enumerate(Train_loader2):
        images = images.to(device)
        labels = labels.to(device)
    
        #forward pass
        outputs = model2(images)
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
    model2.eval()
    val_correct = 0
    val_total=0
    val_loss_epoch = 0.0
    with torch.no_grad():
        for images_val, labels_val in Val_loader2:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = model2(images_val)
            loss_val = criterion(outputs_val, labels_val)

            val_loss_epoch += loss_val.item() * images_val.size(0)
            _, predicted_val = torch.max(outputs_val.data, 1)
            val_total += labels_val.size(0)
            val_correct += (predicted_val == labels_val).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    val_loss_epoch /= val_total
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss_epoch)

    file_name = f'ResNet_cifar10_cat_ship_dog_epoch{epoch+1}.pth'
    full_path = os.path.join(save_folder2, file_name)
    torch.save(model2.state_dict(), full_path) # Provide the model's state_dict and a file path
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_epoch:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss_epoch:.4f}, Val Accuracy: {val_accuracy:.2f}%')
  
#4) test 
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in Test_loader2:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model2(images)
        
        #value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        
    acc = 100* n_correct/n_samples
    print(f'accuracy = {acc}')
    print(n_samples)
'''
#print acc
'''
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
'''

#sử dụng model cho GradCAM
path='C:/Users/THE ANH/Desktop/code/AI/deep_learning/Computer_vision/hist/hist_ResNet/model1/ResNet_cifar10_cat_ship_epoch50.pth'

from transformers import ResNetForImageClassification
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')


for i, (image, label) in enumerate(Test_loader1):
    batch_data, batch_label = image, label
    break
single_image = batch_data[1]
print(single_image.shape)
print(batch_label[0])

if single_image.is_cuda:
    image_on_cpu = single_image.cpu()
else:
    image_on_cpu = single_image

# 2. Chuyển thứ tự các chiều từ (C, H, W) sang (H, W, C)
image_permuted = image_on_cpu.permute(1, 2, 0)

# 3. Chuyển Tensor sang NumPy Array
image_np = image_permuted.numpy()

# 4. Chuyển đổi kiểu dữ liệu và giá trị (nếu tensor là float trong khoảng [0, 1])
# Kiểm tra kiểu dữ liệu
if image_np.dtype == np.float32 or image_np.dtype == np.float64:
    # Giả sử giá trị trong khoảng [0, 1], nhân 255 và chuyển sang uint8
    # Cần đảm bảo giá trị nằm trong khoảng [0, 1] trước khi nhân 255
    image_np = (image_np * 255).astype(np.uint8)
    # Có thể cần clip giá trị để đảm bảo nằm trong [0, 255] nếu quá trình trước đó có thể tạo ra giá trị ngoài khoảng
    # image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

# Hiển thị ảnh trong cửa sổ
scale_factor = 10 # Muốn phóng to gấp 4 lần
new_width = int(image_np.shape[1] * scale_factor) # Lấy chiều rộng gốc (index 1)
new_height = int(image_np.shape[0] * scale_factor) # Lấy chiều cao gốc (index 0)
new_size = (new_width, new_height) # Kích thước mới dưới dạng (width, height)

# Hoặc đơn giản hơn, sử dụng fx và fy:
fx = scale_factor
fy = scale_factor

# Sử dụng cv2.resize() với phương pháp nội suy (ví dụ: INTER_CUBIC)
# Khi dùng fx, fy, đặt dsize=None hoặc (0, 0)
resized_image_np = cv2.resize(image_np, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

cv2.imshow("Single Image", resized_image_np)

# Chờ người dùng nhấn một phím bất kỳ
cv2.waitKey(0)

# Đóng tất cả các cửa sổ OpenCV
cv2.destroyAllWindows()

#plt.show(single_image)
img_tensor = transforms.ToTensor()(image_np) 

def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]

#we will show GradCAM for the "Egyptian Cat" and the 'Remote Control" categories:
targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, "cat")),
                       ClassifierOutputTarget(category_name_to_index(model, "ship"))]

target_layer = model.resnet.encoder.stages[-1].layers[-1]



try:
    display(run_grad_cam_on_image(model=model,
                      target_layer=target_layer,
                      targets_for_gradcam=targets_for_gradcam,
                      reshape_transform=None,
                      input_image=image,
                      input_tensor=img_tensor))  
    
except Exception as e:
    print(f'không thể display 1, lỗi {e}')

