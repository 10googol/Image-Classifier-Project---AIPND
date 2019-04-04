import time
import copy
import os
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image

###########################################################
################Begin Function Definition##################

def args_parser():
    #######################################################################
    # Command Line Argument parser
    #######################################################################
    parser = argparse.ArgumentParser(description='trainer file')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture')
    parser.add_argument('--hidden_units', type=int, default=500, help='hidden units for layer')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to a file')
    args = parser.parse_args()
    return args


def process_data(train_dir, test_dir, valid_dir):
    #######################################################################
    # Setup transforms and datasets then load the testing validation and
    # training datasets for processing.
    #######################################################################
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize])

    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True, drop_last = False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last = False)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle = True, drop_last = True)

    return trainloader, testloader, validloader, train_datasets


def pretrained_network(arch):
    #######################################################################
    # Load a Pytorch provided pretrained network. This function loads the
    # vgg16 by default as no other pretrained network is supported by this
    # script at this time.
    #######################################################################
    if arch == None or arch == 'vgg':
        model = models.vgg16(pretrained=True)
        print('Use vgg16')
    else:
        print('Defaulting model to vgg16 - no other pretrained CNN supported at this time')
        model = models.vgg16(pretrained=True)

    return model


def set_classifier(model, hidden_units):
    #######################################################################
    # Setup a new classifier to replace the one provided in the Pytorch
    # pretrained network. There are 25088 inputs and 102 output which are
    # the predicted probablities of the flower categories.
    #######################################################################
    for param in model.parameters():
        param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hidden_units)),
                                  ('relu1', nn.ReLU()),
                                  ('d1', nn.Dropout(p=0.2)),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1)),
                                  ]))

    model.classifier = classifier
    return model


def train_model(epochs, trainloader, validloader, gpu, model, optimizer, criterion, train_datasets ):
    #######################################################################
    # The train_model uses loaded datasets to train a model to classify
    # 102 categories of flowers. The loaded datasets are trainloader and
    # validloader.
    #######################################################################
    best_weights = copy.deepcopy(model.state_dict())
    if gpu:
        model.to('cuda')
    else:
        print("Model training will be ran on the local CPU.")

    start = time.time()

    for e in range(epochs):
        train_loss = 0
        train_accuracy = 0
        best_accuracy = 0
        model.train()
        for ii, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()

            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')

            output = model.forward(images)
            _, preds = torch.max(output, 1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss
            ps = torch.exp(output).data
            train_corrects = (labels.data == ps.max(1)[1])
            train_accuracy += train_corrects.type_as(torch.FloatTensor()).mean()

        model.eval() #Prepare the model for validation by placing it in eval mode
        valid_loss, valid_accuracy = validate_model(model, validloader, gpu, optimizer, criterion) #Validate the model after a training pass/epoch

        print("Epoch: {}/{} ".format(e+1, epochs))
        display_model_stats(train_loss, train_accuracy, valid_loss, valid_accuracy, trainloader, validloader)

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_weights = copy.deepcopy(model.state_dict())

    print(f"Device = cuda; Time per batch: {(time.time() - start)/3:.3f} seconds")

    model.load_state_dict(best_weights)

    model.class_to_idx = train_datasets.class_to_idx

    return model


def validate_model(model, testloader, gpu, optimizer, criterion):
    #######################################################################
    # validate_model function compares current training pass with a run
    # through a different set of data. The model is set to eval in this
    # function and only a forward pass is done; no backprop.
    #######################################################################
    if gpu:
        model.to('cuda:0')
    else:
        print("Model validation will be ran on the local CPU.")

    valid_accuracy = 0
    valid_loss = 0
    for ii, (images2,labels2) in enumerate(testloader):
        optimizer.zero_grad()
        if gpu:
            images2, labels2 = images2.to('cuda:0') , labels2.to('cuda:0')

        with torch.no_grad():
            outputs = model.forward(images2)
            loss = criterion(outputs,labels2)
            ps = torch.exp(outputs).data
            valid_loss += loss
            valid_corrects = (labels2.data == ps.max(1)[1])
            valid_accuracy += valid_corrects.type_as(torch.FloatTensor()).mean()

    return valid_loss, valid_accuracy


def display_model_stats(train_loss, train_corrects, valid_loss, valid_accuracy, trainloader, validloader):
    #######################################################################
    # This function caculates train and valid loss and accuracy and then
    # displays that information to the screen as the model is training.
    #######################################################################
    print("******************************************")
    print("Train Loss: {:.4f}".format(train_loss/len(trainloader)),
          "Train Accuracy: {:.4f}".format(train_corrects.double()/len(trainloader)))
    print("Valid Loss: {:.4f}".format(valid_loss/len(validloader)),
          "Valid Accuracy: {:.4f}\n".format(valid_accuracy/len(validloader)))


def save_chkpoint(model, optimizer, args):
    #######################################################################
    # Save a checkpoint of the currently trained model for future use in
    # the predict script.
    #######################################################################
    model.cpu()
    torch.save({'arch': 'vgg16',
                'epochs': 10,
                'input_size': 25088,
                'output_size': 102,
                'optim_dict': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'classifier': model.classifier,
                'class_to_idx': model.class_to_idx},
                'classifier.pth')


def load_chkpoint(checkpoint):
    #######################################################################
    # Load checkpoint is used in the predict script to load the trained
    # model's state_dict, optimizer state_dict, class_to_idx and various
    # saved hyperparameters.
    #######################################################################
    model = models.vgg16(pretrained=True)
    checkpoint = torch.load(checkpoint)

    for param in model.parameters():
        param.requires_grad = False

    model.optimizer = checkpoint['optim_dict']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint["epochs"]
    model.class_to_idx = checkpoint['class_to_idx']


    model.eval()

    return model

################End Function Definition##################
#########################################################

def main():
    #######################################################################
    # main program function
    #######################################################################
    args = args_parser()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    trainloader, testloader, validloader, train_datasets = process_data(train_dir, test_dir, valid_dir)
    print(args.epochs)
    model = pretrained_network(args.arch)

    for param in model.parameters():
        param.requires_grad = False

    model = set_classifier(model, args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    train_model2 = train_model(args.epochs, trainloader, validloader, args.gpu, model, optimizer, criterion, train_datasets)
    save_chkpoint(train_model2, optimizer, args.save_dir)
    print('Completed!')


if __name__ == '__main__': main()
