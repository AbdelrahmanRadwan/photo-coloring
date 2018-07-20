import os
from flask import (Flask,
                   render_template,
                   request)

from temp import dummyfun

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def delete_images(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e) #Rabna ma ygyb Erorr

@app.route("/")
def index():
    #delete_images(os.path.join(APP_ROOT, 'static/'))
    return render_template("upload.html")


@app.route("/upload", methods=['POST','GET'])
def upload():
    global destination
    target = os.path.join(APP_ROOT, 'static/')
    # delete_images(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)

#    print(filename)
    dummyfun(destination)
    return render_template("complete.html", value=filename)


@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('static/')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)


if __name__ == "__main__":
    app.run()
