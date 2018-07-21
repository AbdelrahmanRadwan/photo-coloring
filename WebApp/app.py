import os
from flask import (Flask,
                   render_template,
                   request)

from temp import dummy_fun

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
        except Exception as e:
            print(e)


@app.route("/")
@app.route('/Home')
@app.route('/home')
def index():
    """
    :return: Render the main Page
    """
    delete_images(os.path.join(APP_ROOT, 'images/'))
    return render_template("upload.html")


@app.route("/upload", methods=['POST'])
def upload_image():
    """
    Upload Images  to the Static Folder
    :return:
    """
    global destination
    target = os.path.join(APP_ROOT, 'images/')
    # delete_images(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)

    dummy_fun(target+filename)
    return render_template("complete.html", value=filename)


@app.route('/gallery')
def display_all_images():
    """
    :return: Render all Images in Static Folder
    """
    image_names = os.listdir('images/')
    Images = list()
    for i in range(len(image_names)):
        Images.append('/home/mostafa/photo-coloring/WebApp/images/'+str(image_names[i]))
    print(Images)
    return render_template("gallery.html", image_names=Images)


if __name__ == "__main__":
    app.run()
