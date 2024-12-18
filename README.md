# LSML2-SGA
The final project for LSML-2 course

# Project description
The project is realized via Python API with use of Flask module. 

The server is designed to classify Simpson characters with use of convolutional neural network. For this application pre-trained ResNet50 was used. To fine-tune the model for this specific task, layer 4 and last fully connected layer were trained. The resulting model demonstrasted f1-score equal to 0.971

The model, used in the application, couldn't be uploaded to this repository due to memory restriction, but it can be found via the [link](https://drive.google.com/file/d/1daGyT5TqYlaiC09lSAupiUluzLpZsTXM/view?usp=sharing) or in file [link_to_model.txt](https://github.com/valeriiaso/LSML2-SGA/blob/main/link_to_model.txt)

The [general_notebook.ipynb](https://github.com/valeriiaso/LSML2-SGA/blob/main/general_notebook.ipynb) contains the overall process of data loading and model training and evaluation.

The [app.py](https://github.com/valeriiaso/LSML2-SGA/blob/main/app.py) contains the code for the Flask application itself.
