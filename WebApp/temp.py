from PIL import Image
def dummyfun(img):
    print("Hello from a Other File")
    X = Image.open(img)
    X.save("static/tfffffffsc", "JPEG")