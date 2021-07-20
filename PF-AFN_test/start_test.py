from PIL import Image
from ia_init import ia_init
from io import BytesIO
import base64

test_img = Image.open("./dataset/test_clothes/003434_1.jpg")
buffered = BytesIO()
test_img.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue())
ia_init(img_str, img_str)