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

## Future Work
- Training, Training and more Training
- Train on more data clusters
- Extend the project to be able to colorize Videos

## References
- http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/
