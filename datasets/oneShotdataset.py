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
#from watch import NlabelTovector
import getpass  
userName = getpass.getuser()

pathminiImageNet = '/home/'+userName+'/data/miniImagenet/'
pathImages = os.path.join(pathminiImageNet,'images/')
# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x)


class miniImagenetOneshotDataset(data.Dataset):
    def __init__(self, dataroot = '/home/'+userName+'/data/miniImagenet', type = 'train',ways=5,shots=1,test_num=1,epoch=100):
        # oneShot setting
        self.ways = ways
        self.shots = shots
        self.test_num = test_num # indicate test number of each class
        self.__size = epoch

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

    def __getitem__(self, index):

        supportFirst = True
        supportImages = 1
        supportBelongs = torch.LongTensor(self.ways*self.shots,1)
        testFirst = True
        testImages = 1
        testBelongs = torch.LongTensor(self.ways*self.test_num,1)

        selected_classes = np.random.choice(self.data.keys(), self.ways, False)
        for i in range(self.ways):
            files = np.random.choice(self.data[selected_classes[i]], self.shots, False)
            for j in range(self.shots):
                image = self.transform(os.path.join(pathImages,str(files[j])))
                image = image.unsqueeze(0)
                if supportFirst:
                    supportFirst=False
                    supportImages = image
                else:
                    supportImages = torch.cat((supportImages,image),0)
                supportBelongs[i*self.shots+j,0] = i
            files = np.random.choice(self.data[selected_classes[i]], self.test_num, False)
            for j in range(self.test_num):
                image = self.transform(os.path.join(pathImages,str(files[j])))
                image = image.unsqueeze(0)
                if testFirst:
                    testFirst = False
                    testImages = image
                else:
                    testImages = torch.cat((testImages,image),0)
                testBelongs[i*self.test_num+j,0] = i

        return supportImages,supportBelongs,testImages,testBelongs

    def __len__(self):
        return self.__size

'''
dataTrain = torch.utils.data.DataLoader(miniImagenetOneshotDataset(type='train',ways=5,shots=5,test_num=15,epoch=100),batch_size=1,shuffle=False,num_workers=4)


#for j in range(5):

for i,(supportInputs,supportLabels,supportBelongs,testInputs,testLabels,testBelongs) in tqdm(enumerate(dataTrain)):
    haha = 1
    if i<=5:
        print(i,supportInputs.size(),supportLabels.size(),supportBelongs.size(),testInputs.size(),testLabels.size(),testBelongs.size())
    #print(testLabels)
    if i==0:
        print(supportLabels[0,0:3])
'''


