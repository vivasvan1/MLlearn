This is an AlexNet implementation for classification of 1000 class (thought provided data is only 3 class if supplied with enough class data u can expand the range of output)

No this model has not completed its training session(not pretrained) but I have verified that the training accuratly works without any error(i.e. running the code for about 100 steps i ensured that the loss decreases).

To execute the code:

Run the following line in terminal:
python3 imple.py

To create more training data for the model:
1) Download images of some dog breed or anything else and put those images in a folder same as this file.
2) Open the scaleAndShapeData.py and add the folder name to the list as shown in comments of that folder.
3) Open terminal and execte scaleAndShapeData.py

now execte the imple.py file to extend the training of the previous trained model.