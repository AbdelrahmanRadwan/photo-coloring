from PIL import Image
import os


def duplicate_img(img):
    """
    :param img: image Path
    :return: Save an image duplicate in static folder
    """
    x = Image.open(img)
    real_name = os.path.basename(img)
    x.save("static/pics/{}".format("Dup"+real_name), "JPEG")
