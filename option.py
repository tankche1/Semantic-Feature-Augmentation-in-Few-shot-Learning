
import argparse
import os

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Tank Shot')
        parser.add_argument('--batchSize', default=128,type=int,
                            help='Batch Size')
        parser.add_argument('--nthreads', default=4,type=int,
                            help='threads num to load data')
        parser.add_argument('--tensorname',default='resnet18',type=str,
                            help='tensorboard curve name')
        parser.add_argument('--ways', default=5,type=int,
                            help='number of class for one test')
        parser.add_argument('--shots', default=5,type=int,
                            help='number of pictures of each class to support')
        parser.add_argument('--test_num', default=15,type=int,
                            help='number of pictures of each class for test')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
