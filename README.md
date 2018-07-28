# Photo Coloring Using End2end CNN based Model!

![Colored](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/10.png  "Colored")

Colouring photos could bring you a lot of fun to your daily life. Just imagine if you coloured your grandma’s old photo and showed it to your family, how surprised and happy they would be.

![Example 2](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/3.jpg  "Example 2")

Colouring a picture with Photoshop is a so exhausting process, so we implemented this Photo coloring tool, which is based on
[Let there be Color](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf), it's Deep Learning based on CNN, 
it can color any grayscale picture with any color scale and picture resolution.

💫 **Version 0.1 out now!**

📖 Documentation
================
## How to Run
**Install the requirements:**
```bash
pip3 install -r requirements.txt 
```
**Running the web app locally**
```bash
python3 WebApp/app.py
```
Then the endpoint will be running on http://127.0.0.1:5000/

![UI](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/20.jpg  "UI")
## How to use the Pre-Trained model
- Download the Pre-trained model from here: 
- Unzip the file into ```algorithm/Models/```
- Run the Run the Endpoint again

## Results

The results are not bad at all! a lot of test cases gonna be so realistic, but the model still needs more training

 Original Image (grayscale)                                                                                                       | Resultant Image (colored)
:-------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:
![Example 1-1](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example1-1.jpg  "Example 1-1") | ![Example 1-2](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example1-2.jpg  "Example 1-2")
![Example 2-1](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example2-1.jpg  "Example 2-1") | ![Example 2-2](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example2-2.jpg  "Example 2-2")
![Example 3-1](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example3-1.jpg  "Example 3-1") | ![Example 3-2](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example3-2.jpg  "Example 3-2")
![Example 4-1](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example4-1.jpg  "Example 4-1") | ![Example 4-2](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example4-2.jpg  "Example 4-2")
![Example 5-1](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example5-1.jpg  "Example 5-1") | ![Example 5-2](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example5-2.jpg  "Example 5-2")
![Example 6-1](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example6-1.jpg  "Example 6-1") | ![Example 6-2](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/Example6-2.jpg  "Example 6-2")

## Paper
This project is an implementation of the [Let there be Color](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf), published 2016.

## Dataset
- Dataset used is [MIT Places Scence Dataset](http://places.csail.mit.edu/).
**Sample of the data used**

![Data Sample](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/8.png  "Data Sample")

- The pre-trained model is trained only on Landscapes, seas and gardens pictures.

## Model Used
![Model](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/7.png  "Model")

## Experiments
![Experiments](https://github.com/AbdelrahmanRadwan/photo-coloring/blob/master/documentation/pics/9.png  "Experiments")


## References & Tutorials

### Useful materials
- Introduction to the problem: http://tinyclouds.org/colorize/
- More details about the solution: http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/

### Other related projects
- https://github.com/pavelgonchar/colornet
- https://github.com/satoshiiizuka/siggraph2016_colorization
- https://github.com/OmarSayedMostafa/Deep-learning-Colorization-for-visual-media

### Tutorials
- CNN: http://cs231n.github.io/convolutional-networks/ 
- CNN: https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html 
- CNN: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/ 
- CNN: https://www.youtube.com/watch?v=Gu0MkmynWkw 
- CNN: https://www.youtube.com/watch?v=2-Ol7ZB0MmU 
- CNN: https://www.youtube.com/watch?v=s716QXfApa0
- Image Colorization: https://www.youtube.com/watch?v=2IWhh1gd_p0
- IMage Colorization: https://www.youtube.com/watch?v=S1HE9SpZl8A
- Image Colorization: https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/
- Image Colorization: https://www.youtube.com/watch?v=tknFQULAAs0
- Image Colorization: https://www.youtube.com/watch?v=KO7W0Qq8yUE
- Deep learning: https://www.youtube.com/watch?v=He4t7Zekob0 

### Shortcuts
- YUV: https://en.wikipedia.org/wiki/YUV 
- CNN: https://en.wikipedia.org/wiki/Convolutional_neural_network 

### CNN practice
- https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8 
- http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/ 
- http://richzhang.github.io/colorization/ 
- https://github.com/microic/niy/tree/master/examples/colorizing_photos
- https://www.kaggle.com/preslavrachev/wip-photo-colorization-using-keras/notebook
- https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/
- https://gist.github.com/standarderror/43582e9a15038806da8a846903438ebe
- https://github.com/coolioasjulio/Cat-Dog-CNN-Classifier
- https://gist.github.com/standarderror/43582e9a15038806da8a846903438ebe
- https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8 https://medium.com/@parthvadhadiya424/hello-world-program-in-keras-with-cnn-dog-vs-cat-classification-efc6f0da3cc5

### Other related papers
- http://richzhang.github.io/colorization/

### Useful Articles
- https://hackernoon.com/colorising-black-white-photos-using-deep-learning-4da22a05f531
- http://whatimade.today/two-weeks-of-colorizebot-conclusions-and-statistics/

### Video colorization
- https://www.youtube.com/watch?v=SM9YwN_Dvv0
- https://www.youtube.com/watch?v=MfaTOXxA8dM

## Future Work
- Training, Training and more Training
- Train on more data clusters
- Extend the project to be able to colorize Videos


