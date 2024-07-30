import numpy as np
import itertools
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time

from dataset import WCamDataset
from wcamnet import WCAMNet

def train(train_path, val_path, save_path = None, name = None, l_rate = None,
        wd = None):
    # function for training WCamNet based on provided data
    
    if name is None:
        name = str(int(time.time()))

    # establish data augmentation of image data for training
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
        transforms.RandomRotation((-45, 45)),
        transforms.Resize((602, 602)),
        transforms.RandomCrop((602,602), 64),
        transforms.ToTensor(),
        transforms.Normalize((0.40560082, 0.41092141, 0.38840549),
            (0.22153167, 0.22203481, 0.22527725)) 
    ])
    
    # data transforms for testing
    test_transforms = transforms.Compose([
        transforms.Resize((602, 602)),
        transforms.ToTensor(),
        transforms.Normalize((0.40560082, 0.41092141, 0.38840549),
            (0.22153167, 0.22203481, 0.22527725)) 
    ])
    
    # use cuda
    device = torch.device("cuda")
    # initialise WCamNet
    net = WCAMNet()
    net.to(device)

    # initialise training data
    trainset = WCamDataset(train_path, train_transforms)
    trainloader = torch.utils.data.DataLoader(dataset = trainset, 
        batch_size = 4, shuffle = True)

    # initialise validation data
    testset = WCamDataset(val_path, test_transforms)
    testloader = torch.utils.data.DataLoader(dataset = testset, 
        batch_size = 4, shuffle = False)

    # configure network optimization and learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr = l_rate, momentum = 0.9, 
            weight_decay = wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5)

    # train for 15 epochs
    net.train()
    for epoch in range(1,2):
        print(epoch)
        for i, (img, label) in enumerate(trainloader):
            img, label = img.to(device), label.to(device) 
            optimizer.zero_grad()
            output = net(img)
            # compute loss and update weights
            label = torch.unsqueeze(label, 1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
    # compute errors on validation set
    net.eval()
    with torch.no_grad():
        # list for saving errors
        test_error = []
        # loop through validation data
        for i, (img, label) in enumerate(testloader):
            img, label = img.to(device), label.to(device) 
            output = net(img)
            # format network output and label, compute error
            label = torch.unsqueeze(label, 1)
            output_np = output.cpu().numpy()
            label_np = label.cpu().numpy()
            abs_error = np.abs(output_np - label_np)
            for j in range(len(abs_error)):
                test_error.append(abs_error[j])
        
        # compute error metrics from stored errors
        test_error = np.array(test_error)
        mae = np.mean(test_error)
        mse = np.mean(test_error**2)
        rmse = np.sqrt(mse)
        print("MAE:", mae)
        print("RMSE:", rmse)
        
        # save weights
        if save_path is not None:
            torch.save(net.state_dict(), save_path + name + ".pth")
            with open(save_path + "results.csv", 'a') as f:
                f.write(name + ',' + str(mae) + ',' + str(rmse) + '\n')
                

if __name__ == "__main__":
    torch.manual_seed(1)
    # argument parsing for initiating training
    ap = argparse.ArgumentParser()
    ap.add_argument("-tr", "--train", help="path to training file", required = True)
    ap.add_argument("-v", "--val", help="path to validation file", required = True)
    ap.add_argument("-lr", "--l_rate", help="learning rate", required = True)
    ap.add_argument("-wd", "--weight_decay", help="learning rate", required = True)
    ap.add_argument("-s", "--save", help="path to directory for saving models and results")
    ap.add_argument("-n", "--name", help="name of the trained model")
    args = vars(ap.parse_args())
    print(args)
    train(args["train"], args["val"], args["save"], args["name"], float(args["l_rate"]), 
            float(args["weight_decay"]))
