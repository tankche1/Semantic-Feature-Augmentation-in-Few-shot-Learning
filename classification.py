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
from datasets import miniImageEmbedding
from torch.optim import lr_scheduler
import copy
import time
rootdir = os.getcwd()

args = Options().parse()


image_datasets = {x: miniImageEmbedding.miniImagenetEmbeddingDataset(type=x)
                  for x in ['train', 'val','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batchSize,
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

classificationNetwork = ClassificationNetwork().cuda()

#############################################
#Define the optimizer

criterion = nn.CrossEntropyLoss()

optimizer_embedding = optim.Adam([
                {'params': classificationNetwork.parameters()},
            ], lr=0.001)

embedding_lr_scheduler = lr_scheduler.StepLR(optimizer_embedding, step_size=10, gamma=0.5)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [ 'train']:
            
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            

            running_loss = 0.0 
            tot_dist = 0.0
            running_corrects = 0
            loss = 0

            # Iterate over data.
            for i,(inputs,labels) in tqdm(enumerate(dataloaders[phase])):

                c = labels
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                _, preds = torch.max(outputs.data.cpu(), 1)

                labels = labels.view(labels.size(0))

                loss = criterion(outputs, labels)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == c.squeeze(1))


            epoch_loss = running_loss / (dataset_sizes[phase]*1.0)
            epoch_acc = running_corrects / (dataset_sizes[phase]*1.0)
            info = {
                phase+'loss': running_loss,
                phase+'Accuracy': epoch_acc,
            }

            print('{} Loss: {:.4f} Accuracy: {:.4f} '.format(
                phase, epoch_loss,epoch_acc))

            # deep copy the model
            if phase == 'train' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        
        print()
        if epoch%30 == 0 :
            torch.save(best_model_wts,os.path.join(rootdir,'models/'+str(args.tensorname)+'.t7'))
            print('save!')
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


classificationNetwork = train_model(classificationNetwork, criterion, optimizer_embedding,
                         embedding_lr_scheduler, num_epochs=100)


torch.save(classificationNetwork.state_dict(),os.path.join(rootdir,'models/'+str(args.tensorname)+'.t7'))


