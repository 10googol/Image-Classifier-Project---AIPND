import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import train

###########################################################
################Begin Function Definition##################

def args_parser():
    #######################################################################
    # Command Line Argument parser
    #######################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image to predict')
    parser.add_argument('--checkpoint', type=str, default='classifier.pth', help='Model checkpoint to use when predicting')
    parser.add_argument('--topk', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', default='cat_to_name.json', type=str, help='JSON file containing label names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    return args


def predict(image_path, model, topk):
    #######################################################################
    # Predict function processes input image for loaded model checkpoint
    # then takes a pass through the model provided to determine image
    # category.
    #######################################################################
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()

    output = model.forward(img) #Pass image through model
    probability = F.softmax(output.data,dim=1) #get prediction probablities

    return probability.topk(topk)

def process_image(image):
    #######################################################################
    # Called from the predict function the process_image function performs
    # the necessary work to prepare the image for a pass throught the loaded
    # model.
    #######################################################################
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = Image.open(image)
    img = preprocess(img).float()

    return img


def convert_labels_probs(flower_class, probability, model):
    #######################################################################
    # convert_labels_probs function is used to convert labels and probst
    # to numpy arrays to prepare them for display. This function is called
    # from the main program.
    #######################################################################
    top_labs = flower_class.detach().numpy().tolist()[0]
    top_probs = probability.detach().numpy().tolist()[0]
    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]

    return top_labels, top_probs

################End Function Definition##################
#########################################################


def main():
    #######################################################################
    # main program function
    #######################################################################
    args = args_parser()

    with open(args.category_names, 'r') as f:
         cat_to_name = json.load(f)

    model = train.load_chkpoint(args.checkpoint)
    probability, flower_class = predict(args.image, model, args.topk)
    top_labs, top_probs = convert_labels_probs(flower_class, probability, model)

    # Get matching flower name and append them to the labels2 list for processing
    labels2 = []
    for i in top_labs:
        label = cat_to_name[i]
        labels2.append(label)

    # Parse and format predicted flower names and prediction probablities.
    print('  Predicted Flower Name  |  Prediction Probability  ')
    print('----------------------------------------------------')
    index = 0
    for i in labels2:
        space = 20 - len(i) # Calculate spaces to multiple in print statement
        print('  ', i, space*' ', '|', '  ', '%', (top_probs[index] * 100))
        index += 1

if __name__ == '__main__': main()
