# Resnet18 Baseline



We provided the code to reach our baseline performance in miniImagenet.(Resnet18+SVM)



## Datasets

```
Please put the data in:
/home/yourusername/data/miniImagenet

The images are put in 
.../miniImagenet/images
such as:miniImagenet\images\n0153282900000006.jpg
We provide the data split,please put them at 
.../miniImagenet/train.csv
.../miniImagenet/test.csv
.../miniImagenet/val.csv
```

## Train

If you want to train a resnet18 from scratch by yourself:

```
python train.py
```

You can also used our provided model 

```
/samplecode/models/resnet18.t7
```

Then used it to do the one-shot task:

```
python SVM.py
```

