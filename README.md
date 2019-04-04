A. AI Programming with Python Project - Flower Image Classifier
 ---------------------------------------------------------------------------------------------
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students
first develop code for an image classifier built with PyTorch, then convert it into a command line
application.

Going forward, AI algorithms will be incorporated into more and more everyday applications.
For example, you might want to include an image classifier in a smart phone app. To do this,
you'd use a deep learning model trained on hundreds of thousands of images as part of the overall
application architecture. A large part of software development in the future will be using these
types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers.
You can imagine using something like this in a phone app that tells you the name of the flower your
camera is looking at. In practice you'd train this classifier, then export it for use in your
application. We'll be using this dataset of 102 flower categories, you can see a few examples below.

B. Prerequisites
---------------------------------------------------------------------------------------------
To view this project you will need to have the following software installed on your computer.
1. Anaconda - https://www.anaconda.com/distribution/#download-section
   a. Follow the link to the Anaconda download web site.
   b. Choose your operating system
   c. Install the Python 3.7 version of Anaconda
2. Juypter - Installed with Anaconda
   a. To start use the command line to type:
      jupyter notebook
   b. The jupyter webpage will open automatically from there you can veiw any .ipynb file

 C. Required Project Files
 ---------------------------------------------------------------------------------------------
 1. Image Classifier Project.ipynb - Notebook file for Jupyter. This file has the basic
    model programmed with image output and predictions. Jupyter is used to experiment with
    code ideas and layout the fundamental structure of this project.
 2. train.py - Python script were the basic experimental code is turned into a more usable
    project by making the training portion of the code from the Jupyter notebook available for
    usage from the command line. See below for commmand line argument options.
 3. predict.py - Python script were the basic experimental code is turned into a more usable
    project by making the predict portion of the code from the Jupyter notebook available for
    usage from the command line. The predict script is used after a model has been trained and
    a checkpoint  of that model is available to load. This script takes a directory path and
    image name and runs it throught the model to predict the most likely type of flower
    category. See below for command line argument options.
 4. classifier.pth - This file is the save checkpoint of a trained model it contains the
    model state dict, the optimizer state dict and various save hyperparameters that were used
    during the model training portionn of the script.
 5. cat_to_name.json - The JSON file has the mapping from category label to category name and is
    used as part of the predict script to get the name of the flower the model predict from the
    label.
 6. README.md - A basic overview of the flower Image Classifier project.

 D. Command Line Arguments.
  ---------------------------------------------------------------------------------------------
    1. Train a new network on a data set with train.py
       a. Basic usage: python train.py data_directory
       b. Prints out training loss, validation loss, and validation accuracy as the network
          trains
          Options:
          i.   Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
          ii.  Choose architecture: python train.py data_dir --arch "vgg13"
          iii. Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units
               512 --epochs 20
          iv.  Use GPU for training: python train.py data_dir --gpu
    2. Predict flower name from an image with predict.py along with the probability of that name.
       That is, you'll pass in a single image /path/to/image and return the flower name and class
       probability.
       a. Basic usage: python predict.py /path/to/image checkpoint
       Options:
       i.   Return top K most likely classes: python predict.py input checkpoint --top_k 3
       ii.  Use a mapping of categories to real names: python predict.py input checkpoint
            --category_names cat_to_name.json
       iii. Use GPU for inference: python predict.py input checkpoint --gpu
