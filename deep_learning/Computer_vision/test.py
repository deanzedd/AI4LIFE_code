import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random
import os
from tqdm import tqdm # Cần import tqdm nếu bạn sử dụng nó
from sklearn.metrics.pairwise import cosine_similarity # Dùng sklearn cho Cosine Similarity

# --- Bổ sung: Định nghĩa lớp ResNet ---
# Bạn cần định nghĩa hoặc import lớp ResNet của bạn ở đây
# Giả sử bạn có một lớp ResNet với cấu trúc tương tự ResNet tiêu chuẩn của torchvision,
# nơi có lớp avgpool trước lớp fc cuối cùng.
# Nếu kiến trúc của bạn khác, bạn cần điều chỉnh tên lớp 'embedding_layer_name'
# và cách trích xuất output của lớp đó.

# Dưới đây là một ví dụ về cấu trúc ResNet cơ bản (không đầy đủ, chỉ để minh họa lớp avgpool và fc)
# Bạn cần thay thế nó bằng định nghĩa lớp ResNet thực tế của bạn.
'''
class ResNet(nn.Module):
    def __init__(self, arch, lr, num_classes=10): # Thêm num_classes nếu ResNet của bạn có thể cấu hình số lớp output
        super(ResNet, self).__init__()
        # ... (các lớp conv, batchnorm, relu, block) ...
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Lớp Global Average Pooling
        self.fc = nn.Linear(in_features, num_classes) # Lớp phân loại cuối cùng

    def forward(self, x):
        # ... (forward qua các lớp) ...
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # embedding = x # Output của avgpool có thể là embedding
        x = self.fc(x)
        return x
'''
# ---> Hãy đảm bảo lớp ResNet của bạn được định nghĩa hoặc import ở đây <---
# Ví dụ: from my_resnet_model import ResNet


# --- Bắt đầu phần code bổ sung cho so sánh Similarity ---

print("\n--- Bắt đầu phần so sánh Similarity ---")

# 1. Tải các trọng số mô hình đã huấn luyện
# *QUAN TRỌNG:* Cập nhật đường dẫn tới các file trọng số đã lưu của bạn
# Đảm bảo bạn đã chạy phần huấn luyện (bỏ comment các vòng lặp training) trước khi chạy phần này
path_to_model1_weights = os.path.join(save_folder, 'ResNet_cifar10_cat_ship_epoch50.pth') # Ví dụ: lấy trọng số sau 50 epoch
path_to_model2_weights = os.path.join(save_folder2, 'ResNet_cifar10_cat_ship_dog_epoch50.pth') # Ví dụ: lấy trọng số sau 50 epoch

# Khởi tạo lại các mô hình với số lớp output phù hợp với tập dữ liệu huấn luyện của chúng
# Model 1: 2 lớp ('cat', 'ship')
model1_compare = ResNet(arch=arch_110, lr=learning_rate, num_classes=len(classes1)).to(device)
# Model 2: 3 lớp ('cat', 'dog', 'ship')
model2_compare = ResNet(arch=arch_110, lr=learning_rate, num_classes=len(classes2)).to(device)

print(f"Tải trọng số cho Model 1 từ: {path_to_model1_weights}")
print(f"Tải trọng số cho Model 2 từ: {path_to_model2_weights}")

# Kiểm tra xem file trọng số có tồn tại không
if not os.path.exists(path_to_model1_weights):
     print(f"Lỗi: Không tìm thấy file trọng số cho Model 1 tại {path_to_model1_weights}")
     print("Hãy chắc chắn bạn đã huấn luyện và lưu model 1.")
     exit()
if not os.path.exists(path_to_model2_weights):
     print(f"Lỗi: Không tìm thấy file trọng số cho Model 2 tại {path_to_model2_weights}")
     print("Hãy chắc chắn bạn đã huấn luyện và lưu model 2.")
     exit()

# Tải state dict
model1_compare.load_state_dict(torch.load(path_to_model1_weights, map_location=device))
model2_compare.load_state_dict(torch.load(path_to_model2_weights, map_location=device))

# Chuyển mô hình sang chế độ đánh giá (evaluation mode)
model1_compare.eval()
model2_compare.eval()

print("Đã tải trọng số và chuyển mô hình sang chế độ evaluation.")

# 2. Trích xuất Embedding
# Sử dụng forward hook để lấy output của một lớp cụ thể
def get_layer_output_with_hook(model, input_tensor, layer_name):
    """
    Trích xuất output của một lớp cụ thể trong mô hình sử dụng forward hook.
    Args:
        model (torch.nn.Module): Mô hình PyTorch.
        input_tensor (torch.Tensor): Dữ liệu đầu vào (có batch dimension).
        layer_name (str): Tên của module/lớp bạn muốn trích xuất output.

    Returns:
        torch.Tensor: Output của lớp được chỉ định.
    """
    output = {}
    # Hàm hook sẽ được gọi khi forward pass đi qua module đích
    def hook_fn(module, input, output_tensor):
        # Lưu output của module, sử dụng .detach() để ngắt khỏi computational graph
        output[layer_name] = output_tensor.detach()

    # Tìm module đích theo tên
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"Lớp '{layer_name}' không tìm thấy trong mô hình.")

    # Đăng ký hook
    hook = target_layer.register_forward_hook(hook_fn)

    # Thực hiện forward pass để kích hoạt hook
    with torch.no_grad(): # Không tính gradient trong quá trình trích xuất
        model(input_tensor.to(device))

    # Hủy đăng ký hook sau khi sử dụng
    hook.remove()

    if layer_name in output:
        # Squeeze(0) để loại bỏ batch dimension giả nếu input có batch size 1
        return output[layer_name].squeeze(0)
    else:
         # Trường hợp này không nên xảy ra nếu layer được tìm thấy và hook hoạt động
         raise RuntimeError(f"Hook không lấy được output cho lớp '{layer_name}'.")

# Xác định tên lớp bạn muốn trích xuất embedding từ.
# Đối với ResNet tiêu chuẩn của torchvision, 'avgpool' là lớp Global Average Pooling
# trước lớp Linear cuối cùng. Đây là một lựa chọn phổ biến cho embedding.
# Nếu kiến trúc ResNet của bạn khác, bạn cần kiểm tra tên lớp phù hợp bằng cách:
# for name, module in model1_compare.named_modules(): print(name)
# và chọn tên lớp mong muốn.
embedding_layer_name = 'avgpool' # Thay đổi tên này nếu kiến trúc ResNet của bạn khác

print(f"\nTrích xuất embedding từ lớp '{embedding_layer_name}' của mỗi model.")

# 3. Chọn ảnh 'cat' để so sánh
# Lấy một ảnh 'cat' từ tập Test_datasets chung (đã tải ở đầu code)
cat_image_sample = None
cat_label_idx = Test_datasets.class_to_idx['cat'] # Lấy index của lớp 'cat'
original_cat_class_name = classes[cat_label_idx] # Tên gốc trong CIFAR10

# Tìm ảnh 'cat' đầu tiên trong tập Test_datasets
for img, label in Test_datasets:
    if label == cat_label_idx:
        cat_image_sample = img.unsqueeze(0) # Thêm batch dimension (batch size = 1)
        print(f"\nTìm thấy một ảnh '{original_cat_class_name}' từ tập Test để so sánh.")
        break

if cat_image_sample is None:
    print("\nLỗi: Không tìm thấy ảnh 'cat' trong tập Test_datasets để so sánh.")
    # Thoát hoặc xử lý lỗi nếu không tìm thấy ảnh
    exit()

# 4. Trích xuất embeddings cho ảnh 'cat' từ cả hai mô hình
try:
    # Trích xuất embedding từ Model 1 (huấn luyện trên cat, ship)
    embedding_model1 = get_layer_output_with_hook(model1_compare, cat_image_sample, embedding_layer_name)
    print(f"Embedding từ Model 1 có kích thước: {embedding_model1.shape}")

    # Trích xuất embedding từ Model 2 (huấn luyện trên cat, dog, ship)
    embedding_model2 = get_layer_output_with_hook(model2_compare, cat_image_sample, embedding_layer_name)
    print(f"Embedding từ Model 2 có kích thước: {embedding_model2.shape}")

except ValueError as e:
    print(f"Lỗi khi trích xuất embedding: {e}")
    print("Vui lòng kiểm tra lại tên lớp 'embedding_layer_name' xem có đúng với cấu trúc ResNet của bạn không.")
    exit()
except Exception as e:
    print(f"Lỗi không xác định khi trích xuất embedding: {e}")
    exit()


# 5. Tính toán Độ tương đồng (Cosine Similarity)
# Sklearn's cosine_similarity yêu cầu input là mảng numpy 2D (số_mẫu, số_đặc_trưng)
# Chuyển tensor embedding về CPU và numpy, sau đó reshape
embedding_model1_np = embedding_model1.cpu().numpy().reshape(1, -1)
embedding_model2_np = embedding_model2.cpu().numpy().reshape(1, -1)

# Tính Cosine Similarity
similarity_score = cosine_similarity(embedding_model1_np, embedding_model2_np)[0][0]

print(f"\n--- Kết quả So sánh Embeddings ---")
print(f"Độ tương đồng (Cosine Similarity) giữa embedding của ảnh '{original_cat_class_name}'")
print(f"từ Model 1 (cat, ship) và Model 2 (cat, dog, ship) tại lớp '{embedding_layer_name}': {similarity_score:.4f}")

# --- Diễn giải kết quả ---
print("\n--- Ý nghĩa của điểm tương đồng ---")
print(f"- Điểm số này ({similarity_score:.4f}) cho biết mức độ 'giống nhau' trong cách")
print(f"  Model 1 và Model 2 biểu diễn cùng một ảnh '{original_cat_class_name}' ở mức đặc trưng học được tại lớp '{embedding_layer_name}'.")

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


print("\n--- Lưu ý quan trọng ---")
print("- Kết quả này chỉ dựa trên MỘT MẪU ảnh 'cat' duy nhất và MỘT LỚP cụ thể ('avgpool').")
print("- Để có đánh giá toàn diện hơn, bạn nên:")
print("  - Lặp lại quá trình với nhiều ảnh 'cat' khác nhau (ví dụ: tính trung bình độ tương đồng trên một batch ảnh 'cat' test).")
print("  - So sánh embeddings từ CÁC LỚP khác nhau trong mô hình (đặc biệt là các lớp sâu hơn).")
print("  - So sánh embeddings của các lớp khác ('ship', 'dog') giữa hai mô hình.")
print("  - Sử dụng các metric khoảng cách khác (ví dụ: Euclidean Distance).")
print("  - Phân tích phân phối của embeddings (ví dụ: sử dụng t-SNE hoặc UMAP để visualize không gian embedding).")
print("  - So sánh ranh giới quyết định của hai mô hình.")