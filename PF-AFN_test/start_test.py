from tryon_process import tryon_process
from PIL import Image
from io import BytesIO
import base64


if __name__ == '__main__':
    test_img_cloth = Image.open("./dataset/test_clothes/006026_1.jpg")
    buffered_cloth = BytesIO()
    test_img_cloth.save(buffered_cloth, format="JPEG")
    cloth_img_str = base64.b64encode(buffered_cloth.getvalue())

    test_img_person = Image.open("./dataset/test_img/014834_0.jpg")
    buffered_person = BytesIO()
    test_img_person.save(buffered_person, format="JPEG")
    person_img_str = base64.b64encode(buffered_person.getvalue())

    tryon_process(cloth_img_str, person_img_str)