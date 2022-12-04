import io
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image
import os



# This code will convert the weights form the pretrained models to be able to use them for Python 3


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


################## AlexNet ##################

def bn_relu(inplanes):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True))

def bn_relu_pool(inplanes, kernel_size=3, stride=2):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

class AlexNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, bias=False)
        self.relu_pool1 = bn_relu_pool(inplanes=96)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, padding=2, groups=2, bias=False)
        self.relu_pool2 = bn_relu_pool(inplanes=192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu3 = bn_relu(inplanes=384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu4 = bn_relu(inplanes=384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu_pool5 = bn_relu_pool(inplanes=256)
        # classifier
        self.conv6 = nn.Conv2d(256, 256, kernel_size=5, groups=2, bias=False)
        self.relu6 = bn_relu(inplanes=256)
        self.conv7 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_pool1(x)
        x = self.conv2(x)
        x = self.relu_pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu_pool5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        return x


def byte_convert(model: dict) -> dict:
    new_model = dict()
    for key in model:
        if type(key) == bytes:
            new_key = key.decode("utf-8")
        else:
            new_key = key
        new_model[new_key] = dict()
        if isinstance(model[key], dict):
            new_model[new_key] = byte_convert(model[key])
        else:
            new_model[new_key] = model[key]
    return new_model



def load_model(model_path: str) -> dict:
    return torch.load(model_path, encoding='bytes', map_location=device)


def load_model2(model_path: str) -> nn.Module:
    with open(model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        return torch.load(buffer, map_location=device)


def save_model(model: nn.Module, state_dict: dict) -> None:
    model.load_state_dict(state_dict)
    model.eval()
    torch.save(model, type(model).__name__ + '.pth')


model = AlexNet()
model = load_model('pytorch-models/alexnet.pth')


model.keys()


state_dict = byte_convert(model)['state_dict']


save_model(AlexNet(), state_dict)


def read_img(root, filedir, transform=None):
    # Data loading
    with open(filedir, 'r') as f:
        lines = f.readlines()  
    output = []    
    for line in lines:
        linesplit = line.split('\n')[0].split(' ')
        addr = linesplit[0]
        #target = torch.Tensor([float(linesplit[1])])
        img = Image.open(os.path.join(root, addr)).convert('RGB')

        if transform is not None:
            img = transform(img)
        
        output.append([img, 0])

    return output


def load_model(pretrained_dict, new):
    model_dict = new.state_dict()
    #print(new.state_dict())
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    new.load_state_dict(model_dict)


import numpy as np

def main():
    # net definition 
    net = AlexNet()
    # net = Nets.ResNet(block = Nets.BasicBlock, layers = [2, 2, 2, 2], num_classes = 1).cuda()

    # load pretrained model
    load_model(torch.load('AlexNet.pth', map_location=torch.device('cpu')), net) #load_model('pytorch-models/alexnet.pth')
    # load_model(torch.load('./models/resnet18.pth'), net)

    # evaluate
    net.eval()

    # loading data...
    root = 'C:\\Users\\Lenovo\\Dropbox\\DTU\\advancedProject\\Images'
    #valdir = './data/1/test16morph.txt'
    valdir = 'C:\\Users\\Lenovo\\Dropbox\\DTU\\advancedProject\\bonafide_contents.txt'
    #root = 'C:/Users/Lenovo/Documents/DTU-AP/Multi-Morph/asian/af/asian_female_16'
    #valdir = './data/1/morph16.txt'
    #valdir = './data/1/test1.txt'
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  
    val_dataset = read_img(root, valdir, transform=transform)

    with torch.no_grad():
        #label = []
        pred = []

        for i, (img, target) in enumerate(val_dataset):
            print(i)
            img = img.unsqueeze(0)#.cuda(non_blocking=True)
            #target = target#.cuda(non_blocking=True)
            output = net(img).squeeze(1)
            #label.append(target.cpu()[0])
            pred.append(output.cpu()[0])
            #print(i)

        # measurements
        #label = np.array(label)
        prediction = np.array(pred)
        #correlation = np.corrcoef(label, pred)[0][1]
        #mae = np.mean(np.abs(label - pred))
        #rmse = np.sqrt(np.mean(np.square(label - pred)))
        
    #print('Label: {} - Prediction: {}'.format(label.mean(),pred.mean()))
    #print('Prediction: {}'.format(pred.mean()))
    print('Prediction Array: ', prediction)
    np.save('C:\\Users\\Lenovo\\Dropbox\\DTU\\advancedProject\\alex_bonafide_prediction', prediction)
    #print('Correlation:{correlation:.4f}\t'
    #      'Mae:{mae:.4f}\t'
    #      'Rmse:{rmse:.4f}\t'.format(
    #        correlation=correlation, mae=mae, rmse=rmse))


if __name__ == '__main__':
    main()


