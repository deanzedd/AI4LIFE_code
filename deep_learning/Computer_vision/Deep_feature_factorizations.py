'''
IDEA:
# với GradCAM thì ta có thể tạo ra heatmap để giải thích output that
# correspond with a target function, the target function is usually the score of one of the categories

The heatmap is computed in a way that’s connected to the network output,
with the aim that pixels that get higher values in the attribution,
would correspond with a higher output in the target function.

The different methods use the internal feature representations from the network
= "open up the black box"

But this visualization still leaves some things to be desired:
1. What are the internal concepts the model finds, if any:
Does the network just see the cat head and body together?
Or maybe it detects them as different concepts ?
We heard that neural networks are able to identify high level features like
ears, eyes, faces and legs. But we’re never actually able to see this in the model explanations.

2.Could it be that the body of the cat also pulls the output towards other categories as well?
Just because it contributes to a higher output for one category, it doesn’t mean it doesn’t contribute to other categories as well.
For example, there are many different types of cats.
To take this into account when we’re interpreting the heatmaps,
we would have to carefully look at all the heatmaps and keep track of them.

3. What about the other objects in the image ?
We typically create a heatmap for a specific category target.
In the case of image-net we have 1,000 categories,
so we can’t display 1,000 heatmaps or overlay them together,
that would be too much information. Maybe we could detect the
top scoring categories for the image and create heatmaps only
for them. But what if one of the objects was detected andthen 
the model just wasn’t very confident about it - 
assigning it a low score relative to the other objects.
We would never know about it.

4. How do we merge all the visualizations into a single image: 
In terms of the visualization itself, if we have 10 heatmaps for 10 categories,
we would need to look at 10 different images.
And some of the pixels could get high values in several heatmaps,
for example different categories of cats. This is a lot of information to unpack and not very effecient.


'''

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
from XAI_hugging_face import display


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

def create_labels(concept_scores, top_k=2):
    """ Create a list with the image-net category names of the top scoring categories"""
    imagenet_categories_url = \
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    labels = eval(requests.get(imagenet_categories_url).text)
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]    
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{labels[category].split(',')[0]}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk

model = resnet50(pretrained=True)
model.eval()
print("Loaded model")

from pytorch_grad_cam.utils.image import show_factorization_on_image

def visualize_image(model, img_url, n_components=5, top_k=2):
    img, rgb_img_float, input_tensor = get_image_from_url(img_url)
    classifier = model.fc
    dff = DeepFeatureFactorization(model=model, target_layer=model.layer4, 
                                   computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)
    
    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()    
    concept_label_strings = create_labels(concept_outputs, top_k=top_k)
    visualization = show_factorization_on_image(rgb_img_float, 
                                                batch_explanations[0],
                                                image_weight=0.3,
                                                concept_labels=concept_label_strings)
    
    result = np.hstack((img, visualization))
    
    # Just for the jupyter notebook, so the large images won't weight a lot:
    if result.shape[0] > 500:
        result = cv2.resize(result, (result.shape[1]//4, result.shape[0]//4))
    
    return result

if __name__ == "__main__":
    display(visualize_image(model, 
                            'https://img.srv1.hodine.com/pages/GRP.000001/PID.000000004/images/chomeo-1653530609.jpeg',
                            n_components=4))

    display(visualize_image(model, 
                            'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrYb_IeXJqRInL_C7utfy2R8fxSLGIbDcW2A&s',
                            n_components=2))
    display(visualize_image(model, 
                            "https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/both.png?raw=true"))
    display(visualize_image(model, 
                            "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/tutorials/puppies.jpg"))
    display(visualize_image(model, 
                            "https://th.bing.com/th/id/R.94b33a074b9ceeb27b1c7fba0f66db74?rik=wN27mvigyFlXGg&riu=http%3a%2f%2fimages5.fanpop.com%2fimage%2fphotos%2f31400000%2fBear-Wallpaper-bears-31446777-1600-1200.jpg&ehk=oD0JPpRVTZZ6yizZtGQtnsBGK2pAap2xv3sU3A4bIMc%3d&risl=&pid=ImgRaw&r=0",
                            n_components=2))

    display(visualize_image(model, 
                            "https://image.tienphong.vn/w890/Uploaded/2025/qhj_hiobgobrfc/2018_09_11/0929311460946724_cho_meo_1_fimo_ZABC.jpg",
                            n_components=4))
    display(visualize_image(model, 
                            'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrYb_IeXJqRInL_C7utfy2R8fxSLGIbDcW2A&s',
                            n_components=4))


'''
lỗi ở hàm show_factorization_on_image do ko còn tosring_rgb() module
@_api.deprecated("3.8", alternative="buffer_rgba")
    def tostring_rgb(self):
        """
        Get the image as RGB `bytes`.

        `draw` must be called at least once before this function will work and
        to update the renderer for any subsequent changes to the Figure.
        """
        return self.renderer.tostring_rgb()
        
@_api.deprecated("3.8", alternative="buffer_rgba")
    def tostring_rgb(self):
        return np.asarray(self._renderer).take([0, 1, 2], axis=2).tobytes()

'''



