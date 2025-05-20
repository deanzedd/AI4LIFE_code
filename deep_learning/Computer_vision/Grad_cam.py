from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import numpy as np
import requests
import cv2
import json
import torch
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
from pytorch_grad_cam import GradCAM
from torchvision.models import resnet50
#from Deep_feature_factorizations import get_image_from_url
#from XAI_hugging_face import run_grad_cam_on_image, display

from transformers import ResNetForImageClassification
#model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')

def get_image_from_url(url):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    img = np.array(Image.open(requests.get(url, stream=True).raw))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor

model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]
#input_tensor = # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]
'''
targets_for_gradcam1 = [ClassifierOutputTarget(category_name_to_index(model, "Egyptian cat")),
                       ClassifierOutputTarget(category_name_to_index(model, "beagle"))]
                       
targets_for_gradcam1 = [ClassifierOutputTarget(285),
                       ClassifierOutputTarget(162)]
'''
#231: collie, 805: soccer ball, 285: Egyptian cat, 207: golden retriever, 418: ball point
targets_for_gradcam1 = [ClassifierOutputTarget(207),
                       ClassifierOutputTarget(418),
                       ClassifierOutputTarget(285)]


targets_for_gradcam2 = [ClassifierOutputTarget(285)]
'''
targets_for_gradcam2 = [ClassifierOutputTarget(207),
                       ClassifierOutputTarget(285)]
'''

targets_for_gradcam3 = [ClassifierOutputTarget(805)]
'''
targets_for_gradcam3 = [ClassifierOutputTarget(231),
                       ClassifierOutputTarget(805)]
'''
#target_layer = model.resnet.encoder.stages[-1].layers[-1]

targets_for_gradcam4 = [ClassifierOutputTarget(207),
                        ClassifierOutputTarget(418)]

path1 = 'https://www.cool-mania.vn/data/product/11/e8b94416d150a16e3b106490853c8c.jfif'
#path2 = "https://image.tienphong.vn/w890/Uploaded/2025/qhj_hiobgobrfc/2018_09_11/0929311460946724_cho_meo_1_fimo_ZABC.jpg"
path2 = 'https://img.srv1.hodine.com/pages/GRP.000001/PID.000000004/images/chomeo-1653530609.jpeg'
path3_ball_dog = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrYb_IeXJqRInL_C7utfy2R8fxSLGIbDcW2A&s'
#ảnh chó vàng và quả bóng 
path4 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQF9Z_1Q6nZNy7DU1kakEPmz21VhyyloZQqNXCuPXC4ggF9kDTKTzli6Dy5bgQTEXvP-Q&usqp=CAU'
# Construct the CAM object once, and then re-use it on many images:
#cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)
##                 cam = GradCAM(model=model, target_layers=target_layers)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = [ClassifierOutputTarget(281)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
##     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
##     grayscale_cam = grayscale_cam[0, :]
##     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

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
    img1, rgb_img_float1, input_tensor1 = get_image_from_url(path1)
    img2, rgb_img_float2, input_tensor2 = get_image_from_url(path2)
    img3, rgb_img_float3, input_tensor3 = get_image_from_url(path3_ball_dog)
    img4, rgb_img_float4, input_tensor4 = get_image_from_url(path4)
    
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor3, targets=targets_for_gradcam3)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_float3, grayscale_cam, use_rgb=True)
    display(visualization)
    
    '''
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor1, targets=targets_for_gradcam1)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_float1, grayscale_cam, use_rgb=True)
    display(visualization)
    
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor2, targets=targets_for_gradcam2)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_float2, grayscale_cam, use_rgb=True)
    display(visualization)
    
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor3, targets=targets_for_gradcam3)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_float3, grayscale_cam, use_rgb=True)
    display(visualization)
    
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor4, targets=targets_for_gradcam4)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_float4, grayscale_cam, use_rgb=True)
    display(visualization)
    '''