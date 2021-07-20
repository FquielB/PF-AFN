from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import base64

def ia_init(cloth_image_str, person_image_str):
    #edge_cloth_image = generateEdgeImage(cloth_image)
    generateEdgeImage(cloth_image_str)
    #edge_cloth_image.show()

def generateEdgeImage(image_str):
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

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    open_cv_image *= mask2[:, :, np.newaxis]

    Image.fromarray(open_cv_image).show()

    """ plt.subplot(121), plt.imshow(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    plt.title("grabcut"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(OLD_IMG, cv2.COLOR_BGR2RGB))
    plt.title("original"), plt.xticks([]), plt.yticks([])

    plt.show() """
 