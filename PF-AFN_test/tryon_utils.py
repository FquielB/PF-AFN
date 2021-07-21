from PIL import Image
import cv2
import numpy as np
import base64
from io import BytesIO
import torchvision.transforms as transforms

def image_preparation(cloth_image_str, person_image_str):
    edge_cloth_image = generate_edge_image(cloth_image_str)
    cloth_image = str_to_image(cloth_image_str)
    person_image = str_to_image(person_image_str)
    images_dict = data_preparation(cloth_image, edge_cloth_image, person_image)
    return images_dict


def generate_edge_image(image_str):
    """ This method transforms entry image to black and white image "edge" """    
    img_binary = base64.b64decode(image_str)
    image_np = np.frombuffer(img_binary, dtype=np.uint8)
    open_cv_image = cv2.imdecode(image_np, flags=1)
    
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    
    OLD_IMG = open_cv_image.copy()
    
    mask = np.zeros(open_cv_image.shape[:2], np.uint8)
    
    SIZE = (1, 65)
    
    bgdModle = np.zeros(SIZE, np.float64)

    fgdModle = np.zeros(SIZE, np.float64)
    rect = (1, 1, open_cv_image.shape[1], open_cv_image.shape[0])
    cv2.grabCut(open_cv_image, mask, rect, bgdModle, fgdModle, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

    return Image.fromarray(mask2)
 

def str_to_image(image_str):
    decoded_image_str = base64.b64decode(image_str)
    buffered_image = BytesIO(decoded_image_str)
    image = Image.open(buffered_image)

    return image


def data_preparation(cloth_image: Image, edge_cloth_image: Image, person_image: Image): 
    
    transform = get_transform()
    transform_E = get_transform(method=Image.NEAREST, normalize=False)

    I = person_image.convert('RGB')

    I_tensor = transform(I)

    C = cloth_image.convert('RGB')
    C_tensor = transform(C)

    E = edge_cloth_image.convert('L')
    E_tensor = transform_E(E)

    input_dict = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}
    return input_dict


def get_transform(method=Image.BICUBIC, normalize=True):
    transform_list = []
    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)