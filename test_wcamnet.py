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

def test(weights_path, test_path = None, save_path = None, name = None):
    # this function implements the testing procedure of WCamNet on provided data
    
    if name is None:
        name = str(int(time.time()))

    # image transforms used for formatting the input images
    test_transforms = transforms.Compose([
        transforms.Resize((602, 602)),
        transforms.ToTensor(),
        transforms.Normalize((0.40560082, 0.41092141, 0.38840549),
            (0.22153167, 0.22203481, 0.22527725)) 
    ])

    # set PyTorch to use cuda
    device = torch.device("cuda")
    # create the WCamNet network and load weights
    net = WCAMNet()
    net.to(device)
    net.load_state_dict(torch.load(weights_path))

    # initialise the testing data
    testset = WCamDataset(test_path, test_transforms)
    testloader = torch.utils.data.DataLoader(dataset = testset, batch_size = 4, shuffle = False)

    # network in evaluation mode
    net.eval()
    with torch.no_grad():
        # list for saving errors
        test_error = []
        # loop through testing data
        for i, (img, label) in enumerate(testloader):
            img, label = img.to(device), label.to(device) 
            output = net(img)
            # format network output for error computations
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
    
    # save results
    if save_path is not None:
        with open(save_path + "results.csv", 'a') as f:
            f.write(name + ',' + str(mae) + ',' + str(rmse) + '\n')
                

if __name__ == "__main__":
    torch.manual_seed(1)
    # parse arguments for initiating testing
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weights", help="path to weights file", required = True)
    ap.add_argument("-te", "--test", help="path to test file")
    ap.add_argument("-s", "--save", help="path to directory for saving models and results")
    ap.add_argument("-n", "--name", help="name of the trained model")
    args = vars(ap.parse_args())
    print(args)
    test(args["weights"], args["test"], args["save"], args["name"])

