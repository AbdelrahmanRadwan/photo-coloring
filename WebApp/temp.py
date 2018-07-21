from PIL import Image
import random


def dummy_fun(img):
    """
    :param img: Imgae Path
    :return: Save an image duplicate in static folder
    """
    print("Hello from a Other File")
    X = Image.open(img)
    X.save("images/{}".format(random.randint(0,10000000000)), "JPEG")