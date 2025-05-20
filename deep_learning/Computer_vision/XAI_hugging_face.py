'''
source: https://jacobgil.github.io/pytorch-gradcam-book/HuggingFace.html
'''

'''
Requires: The Class Activation Map family of algorithms get as an input:

A model

An image

A target function that you want to explain. e.g: “Where does the model see a Remote Control in this image?”

A target layer (or layers). The internal Activations from this layer will be used. Let’s correct the previous bullet: It’s actually - “Where does the model see a Remote Control in these activations from our target layer”.

A reshape function that tells how to to trasnlate the activations, to featuer maps of (typically small) 2D images.
'''

# target function
#Logits: raw outputs(after FCN, logits, before Softmax)
# để ổn định số học trong train do ta dùng loss = cross_entropy (log của xs tính thường rất nhỏ nên ta kết hợp cả logits và output của softmax)
'''
The rational is that with the logits you’re looking only for positive evidence of a Remote-Control,
and not for evidence of what makes it not look like a “Cat”.(KEYS: nó giống với việc học đặc trưng thật sự của con mèo
, chứ không phải học các đặc trưng để phân biệt với con mèo)
In some cases it might make sense to use the score after softmax - then you can use ClassifierOutputSoftmaxTarget.
Important: In case you have a batch with several images, you need 1 target for each of the images
'''
#from Grad_cam import *
from functools import partial
import torch

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
        
    def __call__(self, model_output):
        if len(model_output.shape)==1:
            return model_output[self.category]
        
        return model_output[:,self.category]
    
#Reshapes fixed input 12x12
def reshape_transform_vit_huggingface(x):
    #Remove the CLS token (batch * squence_length(token_length) * features)
    # (CLS token ở đầu token nhưng ta chỉ cần spatial data nên to bỏ token đầu đi)
    activation = x[:,1:,:]
    #reshape to 12x12 spatial image
    activation = activation.view(activation.shape[0], 12, 12, activation.shape[2])
    #transpose the features to be in the second coordinate: (batch, features, height, width)
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations
    
#Reshape with vảrying input image size: (batch, squence_length(token_length), features) --> (batch, features, height, width)
def segformer_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
'''
# sử dụng partial để tạo 1 hàm mới từ hàm nào đó đã có
reshape_transform = partial(segformer_reshape_transform_huggingface,
                            width=img_tensor.shape[2]//32,
                            height=img_tensor.shape[1]//32)
'''


def reshape_transform_cvt_huggingface(tensor, model, width, height):
    tensor = tensor[:, 1:, :]
    tensor = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(-1))
    norm = model.layernorm(tensor)
    return norm.tranpose(2, 3).transpose(1, 2)

'''
We want to re-use ClassifierOutputTarget, that expects a tensor.
However the hugging face models often output a dataclass that contains a logits property.
ClassifierOutputTarget would then break since it expects a tensor..
'''
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x).logits
    
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
from typing import List, Callable, Optional
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg') #?????

#download data
dataset=None
try:
    dataset = load_dataset('huggingface/cats-image', trust_remote_code=True)
    print("Dataset 'huggingface/cats-image' đã được tải thành công.")
    print(dataset)
    print(f'data shape: {dataset.shape}')
    #plt.show(dataset)

    # Bạn có thể truy cập các split (ví dụ: 'test') và các phần tử
    first_image_example = dataset['test'][0]
    print("\nThông tin về phần tử đầu tiên trong split 'test':")
    print(first_image_example)

    # Để xem ảnh (cần cài thêm Pillow: pip install Pillow)
    # try:
    #     first_image_example['image'].show()
    # except Exception as e:
    #     print(f"\nKhông thể hiển thị ảnh. Đảm bảo bạn đã cài Pillow. Lỗi: {e}")
    # thư viện Pillow kế thừa từ PIL cho phép xử lí ảnh tốt
    try:
        first_image_example['image'].show()
    except Exception as e:
        print(f"\nKhông thể hiển thị ảnh. Đảm bảo bạn đã cài Pillow. Lỗi: {e}")
        
except Exception as e:
    print(f"Đã xảy ra lỗi khi tải dataset: {e}")
    print("Vui lòng kiểm tra kết nối mạng và tên dataset.")
#dataset = load_dataset('huggingface/cats-image')
image = dataset['test']['image'][0]
img_tensor = transforms.ToTensor()(image) 
# sử dụng partial để tạo 1 hàm mới từ hàm nào đó đã có
reshape_transform = partial(segformer_reshape_transform_huggingface,
                            width=img_tensor.shape[2]//32,
                            height=img_tensor.shape[1]//32)

""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.

"""
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]

""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.
    
"""
'''
sửa lại input tensor --> NOne
def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module=img_tensor,
                          input_image: Image=image,
                          method: Callable=GradCAM):
    """
    Chạy Grad-CAM trên một ảnh sử dụng mô hình Hugging Face.

    Args:
        model (torch.nn.Module): Mô hình Hugging Face.
        target_layer (torch.nn.Module): Lớp mục tiêu để tính Grad-CAM.
        targets_for_gradcam (List[Callable]): Danh sách các mục tiêu lớp (ClassifierOutputTarget).
        reshape_transform (Optional[Callable]): Hàm biến đổi đầu ra lớp mục tiêu (nếu cần).
        input_tensor (img_tensor): Tensor của ảnh đầu vào.
        input_image (Image): Ảnh đầu vào gốc (PIL Image).
        method (Callable): Phương thức CAM muốn sử dụng (mặc định là GradCAM).

    Returns:
        np.ndarray: Mảng NumPy chứa hình ảnh trực quan hóa Grad-CAM (BGR).
                    Nếu có nhiều mục tiêu, các visualization sẽ được nối ngang.
    """
    with method(model=HuggingfaceToTensorModelWrapper(model),
                target_layers=[target_layer],
                reshape_transform=reshape_transform) as cam:
        
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)
        
        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results=[]
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1]//2, visualization.shape[0]//2))
            results.append(visualization)
            
        return np.hstack(results)
  

'''

def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module=None,
                          input_image: Image=image,
                          method: Callable=GradCAM):
    """
    Chạy Grad-CAM trên một ảnh sử dụng mô hình Hugging Face.

    Args:
        model (torch.nn.Module): Mô hình Hugging Face.
        target_layer (torch.nn.Module): Lớp mục tiêu để tính Grad-CAM.
        targets_for_gradcam (List[Callable]): Danh sách các mục tiêu lớp (ClassifierOutputTarget).
        reshape_transform (Optional[Callable]): Hàm biến đổi đầu ra lớp mục tiêu (nếu cần).
        input_tensor (img_tensor): Tensor của ảnh đầu vào.
        input_image (Image): Ảnh đầu vào gốc (PIL Image).
        method (Callable): Phương thức CAM muốn sử dụng (mặc định là GradCAM).

    Returns:
        np.ndarray: Mảng NumPy chứa hình ảnh trực quan hóa Grad-CAM (BGR).
                    Nếu có nhiều mục tiêu, các visualization sẽ được nối ngang.
    """
    with method(model=HuggingfaceToTensorModelWrapper(model),
                target_layers=[target_layer],
                reshape_transform=reshape_transform) as cam:
        
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)
        
        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results=[]
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1]//2, visualization.shape[0]//2))
            results.append(visualization)
            
        return np.hstack(results)
    
def print_top_categories(model, img_tensor, top_k=5):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k:][::-1]
    for i in indices:
        print(f'Predicted class {i}: {model.config.id2label[i]}')


'''

'''
from transformers import ResNetForImageClassification
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')

#we will show GradCAM for the "Egyptian Cat" and the 'Remote Control" categories:
targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, "Egyptian cat")),
                       ClassifierOutputTarget(category_name_to_index(model, "remote control, remote"))]

# The last layer in the Resnet Encoder:
target_layer = model.resnet.encoder.stages[-1].layers[-1]

#def hàm display
def display(image_array: np.ndarray, window_name: str = "Image Display"):
    """
    Hiển thị một mảng NumPy (biểu diễn ảnh) trong một cửa sổ.
    Hoạt động trong môi trường Python script chuẩn (ví dụ: VS Code).

    Args:
        image_array (np.ndarray): Mảng NumPy chứa dữ liệu ảnh.
                                  Thường ở định dạng (height, width, channels).
        window_name (str): Tên của cửa sổ hiển thị ảnh.
    """
    if image_array is None or image_array.size == 0:
        print("Không có dữ liệu ảnh để hiển thị.")
        return

    # --- Thêm các dòng kiểm tra để chẩn đoán lỗi cv2.imshow ---
    print(f"Debugging imshow input:")
    print(f"  Type of image_array: {type(image_array)}")
    if isinstance(image_array, np.ndarray):
        print(f"  Shape of image_array: {image_array.shape}")
        print(f"  Dtype of image_array: {image_array.dtype}")
        print(f"  Min value: {np.min(image_array)}, Max value: {np.max(image_array)}")
    else:
        print("  Input is not a NumPy array.")
    # --- Kết thúc các dòng kiểm tra ---

    # OpenCV mặc định làm việc với định dạng BGR cho ảnh màu.
    # Nếu mảng ảnh của bạn đang ở định dạng RGB (ví dụ từ PIL sau khi chuyển đổi),
    # bạn có thể cần chuyển đổi sang BGR nếu show_cam_on_image trả về RGB.
    # Tuy nhiên, show_cam_on_image sử dụng cv2.addWeighted, thường làm việc tốt
    # với ảnh gốc BGR hoặc RGB và bản đồ nhiệt grayscale.
    # Kết quả của show_cam_on_image thường là BGR nếu use_rgb=True được xử lý đúng trong hàm đó.
    # Nếu ảnh hiển thị sai màu, hãy thử chuyển đổi:
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


    # Hiển thị ảnh trong một cửa sổ mới
    cv2.imshow(window_name, image_array)

    # Đợi người dùng nhấn một phím bất kỳ để đóng cửa sổ
    print(f"Đang hiển thị ảnh '{window_name}'. Nhấn phím bất kỳ để đóng cửa sổ.")
    cv2.waitKey(0)

    # Đóng tất cả các cửa sổ OpenCV
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        display(run_grad_cam_on_image(model=model,
                        target_layer=target_layer,
                        targets_for_gradcam=targets_for_gradcam,
                        reshape_transform=None))  
        
    except Exception as e:
        print(f'không thể display 1, lỗi {e}')

    try:
        display(run_dff_on_image(model=model,
                            target_layer=target_layer,
                            classifier=model.classifier,
                            img_pil=image,
                            img_tensor=img_tensor,
                            reshape_transform=None,
                            n_components=4,
                            top_k=2)) 
        
    except Exception as e:
        print(f'không thể display 2, lỗi {e}')




    print_top_categories(model, img_tensor)






