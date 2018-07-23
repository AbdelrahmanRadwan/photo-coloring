import os
from flask import (Flask,
                   render_template,
                   request,
                   send_from_directory)

import modeltrain.train as testing
from temp import duplicate_img
from PIL import Image
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def delete_images(folder):
    """
    Delete Images in Specific folder given the Path
    :param folder: The Folder Path
    """
    for the_file_path in os.listdir(folder):
        file_path = os.path.join(folder, the_file_path)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


@app.route("/")
@app.route('/Home')
@app.route('/home')
def index():
    """
    :return: Render the main Page
    """
    delete_images(os.path.join(APP_ROOT, 'static/pics/'))
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def upload_image():
    """
    Upload Images to the Static Folder
    :return:
    """
    global destination
    target = os.path.join(APP_ROOT, 'static/pics/')
    delete_images(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)
    filename

    testing.Test(filename)
#    duplicate_img(target+filename)
    return render_template("complete.html", value=filename)




@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static/pics", filename)


@app.route('/gallery')
def get_gallery():
    """
    Display all images in folder
    """
    image_names = os.listdir('static/pics')
    print(image_names)
    return render_template("DisplayAll.html", image_names=image_names)


if __name__ == "__main__":
    app.run()
