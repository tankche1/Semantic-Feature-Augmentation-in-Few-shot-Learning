import os
import numpy as np
import argparse
import torch
import torch.optim as optim
from tqdm import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt
from option import Options
from datasets import oneShotdataset
from torch.optim import lr_scheduler
import copy
import time
import numpy
rootdir = os.getcwd()
import getpass  
userName = getpass.getuser()

args = Options().parse()

image_datasets = {}
image_datasets['train'] = oneShotdataset.miniImagenetOneshotDataset(type='train',ways=args.ways,shots=args.shots,test_num=args.test_num,epoch=600)
image_datasets['val'] = oneShotdataset.miniImagenetOneshotDataset(type='val',ways=args.ways,shots=args.shots,test_num=args.test_num,epoch=600)
image_datasets['test'] = oneShotdataset.miniImagenetOneshotDataset(type='test',ways=args.ways,shots=args.shots,test_num=args.test_num,epoch=600)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=args.nthreads)
              for x in ['train', 'val','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}

######################################################################
# Define the Embedding Network

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()
        self.convnet = torchvision.models.resnet18(pretrained=False)

        num_ftrs = self.convnet.fc.in_features
        self.convnet.fc = nn.Linear(num_ftrs,64)

    def forward(self,inputs):
        outputs = self.convnet(inputs)
        return outputs

class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.resnet = ClassificationNetwork()
        #self.resnet.load_state_dict(torch.load('/home/'+userName+'/code/oneshot/samplecode/models/resnet18.t7'))
        self.resnet.load_state_dict(torch.load('models/resnet18.t7'))
        self.cls = self.resnet.convnet.fc
        self.cls.load_state_dict(self.resnet.convnet.fc.state_dict())

        self.conv1 = self.resnet.convnet.conv1
        self.conv1.load_state_dict(self.resnet.convnet.conv1.state_dict())
        self.bn1 = self.resnet.convnet.bn1
        self.bn1.load_state_dict(self.resnet.convnet.bn1.state_dict())
        self.relu = self.resnet.convnet.relu
        self.maxpool = self.resnet.convnet.maxpool
        self.layer1 = self.resnet.convnet.layer1
        self.layer1.load_state_dict(self.resnet.convnet.layer1.state_dict())
        self.layer2 = self.resnet.convnet.layer2
        self.layer2.load_state_dict(self.resnet.convnet.layer2.state_dict())
        self.layer3 = self.resnet.convnet.layer3
        self.layer3.load_state_dict(self.resnet.convnet.layer3.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.avgpool = self.resnet.convnet.avgpool

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1 = self.layer1(x) # (, 64L, 56L, 56L)
        layer2 = self.layer2(layer1) # (, 128L, 28L, 28L)
        layer3 = self.layer3(layer2) # (, 256L, 14L, 14L)
        layer4 = self.layer4(layer3) # (,512,7,7)
        x = self.avgpool(layer4) # (,512,1,1)
        x = x.view(x.size(0), -1)
        return x

embedding = EmbeddingNetwork().cuda()
#############################################
#Test the Embedding network


#############################################
#Define the optimizer


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as la 


def Evaluate(num_epochs):
    since = time.time()

    for haha in range(1):

       
        # parameter should be set according to the performance on validation set
        model = SVC(C=10)
        # for simplicity only 1 epoch ,delete the part of calculate confidence interval
        for epoch in range(1):

            # Each epoch has a training and validation phase

            for phase in ['test','train','val']:

                running_loss = 0.0 
                running_corrects = 0.0
                total = 0

                # Iterate over data.
                
                embedding.train(False)

                for i in tqdm(range(600)):
                    support_feature,support_belong,test_feature,test_belong = image_datasets[phase].__getitem__(i)

                    support_feature = torch.squeeze(support_feature,0)
                    test_feature = torch.squeeze(test_feature,0)

                    support_feature = embedding(Variable(support_feature.cuda())).data.cpu()
                    test_feature = embedding(Variable(test_feature.cuda())).data.cpu()

                    support_feature = torch.squeeze(support_feature,0).numpy()
                    support_belong = torch.squeeze(support_belong,0).numpy()
                    test_feature = torch.squeeze(test_feature,0).numpy()
                    test_belong = torch.squeeze(test_belong,0).numpy()
                    
                    support_belong = support_belong.ravel()
                    test_belong = test_belong.ravel()

                    model.fit(support_feature,support_belong)

                    Ans = model.predict(test_feature) # array
                    Ans = numpy.array(Ans)

                    
                    running_corrects += (Ans==test_belong).sum()
                    total += test_feature.shape[0]

                
                Accuracy = running_corrects/(total*1.0)
                info = {
                    'Accuracy': Accuracy,
                }


                print('{}: Accuracy: {:.4f} '.format(phase,
                     Accuracy))
                

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    


Evaluate(num_epochs=10)

