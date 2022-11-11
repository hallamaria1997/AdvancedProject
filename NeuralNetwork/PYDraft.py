import io
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################## AlexNet ##################
def bn_relu(inplanes):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True))

def bn_relu_pool(inplanes, kernel_size=3, stride=2):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

class AlexNet(nn.Module):
    def __init__(self, num_classes=5):
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

net = AlexNet()
net.load_state_dict(torch.load('./pytorch-models/cnn_ADAM_schlr.pth'))
net.eval()


def read_img(root, filedir, transform=None):
    # Data loading
    with open(filedir, 'r') as f:
        lines = f.readlines()
    output = []    
    for line in lines:
        linesplit = line.split('\n')[0].split(' ')
        #print(linesplit)
        addr = linesplit[0]
        #target = torch.Tensor([float(linesplit[1])])
        img = Image.open(os.path.join(root, addr)).convert('RGB')

        if transform is not None:
            img = transform(img)
        
        output.append([img, 0]) #ÍBS: Changed target to 0

    return output


def main():
    # net definition 
    #net = AlexNet()
    # net = Nets.ResNet(block = Nets.BasicBlock, layers = [2, 2, 2, 2], num_classes = 1).cuda()

    # load pretrained model
    #load_model(torch.load('./pytorch-models/cnn_ADAM_schlr.pth', map_location=torch.device('cpu')), net) #load_model('pytorch-models/alexnet.pth')
    # load_model(torch.load('./models/resnet18.pth'), net)
    
    # evaluate
    #net.eval()
    
    #model = torch.load('./pytorch-models/cnn_ADAM_schlr.pth')
    #model.eval()
    #net = AlexNet()
    #net.load_state_dict(torch.load('./pytorch-models/cnn_ADAM_schlr.pth'))
    #net.eval()
    # loading data...
    #root = 'C:/Users/Lenovo/Documents/DTU-AP/SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/Images'
    #valdir = './data/1/test1.txt'
    #root = 'C:/Users/Lenovo/Documents/DTU-AP/Multi-Morph/asian/af/asian_female_16'
    #valdir = './data/1/morph16.txt'
    #valdir = './data/1/test1.txt'
    #root = './data/input_test' #'C:/Users/Lenovo/Documents/AdvancedProject/NeuralNetwork/data/morph16'
    root = 'C:\\Users\\Lenovo\\Dropbox\\DTU\\advancedProject\\morph16_all\\morph16_imgs'
    #valdir = './data/1/test16morph.txt'
    valdir = 'C:\\Users\\Lenovo\\Dropbox\\DTU\\advancedProject\\morph16_all\\morph16_filenames.txt'
    
    #test_image_dir = './data/input_test'
    #test_image_filepath = os.path.join(test_image_dir, 'image5.png')
 
    #print(root, valdir)
    
    
    # Pre-process input image
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  
    val_dataset = read_img(root, valdir, transform=transform)
    #print(val_dataset)
    
    #net.eval()
    
    with open('./data/classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
    
    with torch.no_grad():
        label = [] #ÍBS: We don't need the label
        pred = []
        perc = []
        c = []

        for i, (img, target) in enumerate(val_dataset):
            print(i)
            img = img.unsqueeze(0)#.cuda(non_blocking=True)
            output = net(img).squeeze(1)
            pred.append(output.cpu()[0])
            #print(output.shape)
            _, index = torch.max(output, 1)
            percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
            perc.append(percentage[index[0]].item())
            c.append(classes[index[0]])
            
            
        #print(output.shape)
        
        #with open('./data/classes.txt') as f:
        #    classes = [line.strip() for line in f.readlines()]
        
        _, index = torch.max(output, 1)
 
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
 
        print(classes[index[0]],percentage[index[0]].item())
        
        # ÍBS: If you want to see the confidence of the other classes
        _, indices = torch.sort(output, descending=True)
        print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])

        # measurements
        label = np.array(label) #ÍBS: don't need the label only the prediction
        prediction = np.array(c)

    print('Prediction Array: ', prediction)
    np.save('C:\\Users\\Lenovo\\Dropbox\\DTU\\advancedProject\\morph16_all\\morph16_prediction', prediction)


if __name__ == '__main__':
    main()
    




