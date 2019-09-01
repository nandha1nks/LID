'''
Dataset and Dataloader Creation

Given the train_file and test_file containing the fileNo and respective label,
it checks validity and creates tensors, datasets and respective loaders

Batch Size, ValBatchSize NumWorkers, fileNames are important inputs
'''
from torch.utils.data import DataLoader , TensorDataset
import torch
import numpy as np

def dataset(fileName, train = True):
    inputTxt = open(fileName,"r")
    imgs = []
    labels = []
    for line in inputTxt:
        pars = line.split(' ')
        if len(pars) != 2:
            print(line)
            continue
        if np.isnan(int(pars[0])) and np.isnan(int(pars[1])) and int(pars[1])>5 and int(pars[1])<0:
            print('No valid label' + line)
        imgs.append(int(pars[0]))
        labels.append(int(pars[1]))
    return (torch.from_numpy(np.array(imgs)), torch.from_numpy(np.array(labels)))
    
def dataLoaders(config, trainLoader = True):
    
    if trainLoader:
        train = dataset(config['trainFile'],True)
        trainDataset = TensorDataset(train[0],train[1])
        loader = DataLoader(trainDataset, config['batchSize'],
                                 num_workers = config['numWorkers'],
                                 shuffle = True, pin_memory = False)
    else:
        test = dataset(config['testFile'],False)
        testDataset = TensorDataset(test[0],test[1])    
        loader = DataLoader(testDataset, config['valBatchSize'],
                                num_workers = config['numWorkers'], shuffle = False,
                                pin_memory = False)
    
    return loader