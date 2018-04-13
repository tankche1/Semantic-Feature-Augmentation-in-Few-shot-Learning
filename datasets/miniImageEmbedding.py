import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os.path
import csv
import math
import collections
from tqdm import tqdm

import numpy as np
import getpass  
userName = getpass.getuser()


pathminiImageNet = '/home/'+userName+'/data/miniImagenet/'
pathImages = os.path.join(pathminiImageNet,'images/')
# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x)

class miniImagenetEmbeddingDataset(data.Dataset):
    def __init__(self, dataroot = '/home/'+userName+'/data/miniImagenet', type = 'train'):
        if type == 'specialtest':
            type = 'test'
        # Transformations to the image
        if type=='train':
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                ])
        else:
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

        def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]

                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels

        self.miniImagenetImagesDir = os.path.join(dataroot,'images')

        self.data = loadSplit(splitFile = os.path.join(dataroot,type + '.csv'))

        self.type = type
        self.data = collections.OrderedDict(sorted(self.data.items()))
        self.classes_dict = {self.data.keys()[i]:i  for i in range(len(self.data.keys()))} # map NLabel to id(0-99)

        self.Files = []
        self.belong = []

        for c in range(len(self.data.keys())):
            for file in self.data[self.data.keys()[c]]:
                self.Files.append(file)
                self.belong.append(c)
        

        self.__size = len(self.Files)

    def __getitem__(self, index):

        c = self.belong[index]
        File = self.Files[index]

        path = os.path.join(pathImages,str(File))
        images = self.transform(path)
        return images,torch.LongTensor([c])

    def __len__(self):
        return self.__size

# test the dataset
'''
dataTrain = miniImagenetEmbeddingDataset(type='train')
print(len(dataTrain))

for i,(a,b,c) in tqdm(enumerate(dataTrain)):
    if i<4:
        print(i,a.size(),b.size(),c.size())

'''
