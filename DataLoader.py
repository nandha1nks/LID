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
    
def dataLoaders(config):
    train = dataset(config['train_file'])
    trainDataset = TensorDataset(train[0],train[1])
    
    test = dataset(config['test_file'],False)
    testDataset = TensorDataset(test[0],test[1])
    
    trainLoader = DataLoader(trainDataset,config['batch_size'],num_workers=0,shuffle = True,pin_memory = False)
    testLoader = DataLoader(testDataset,config['val_batch_size'],num_workers=0,shuffle = False,pin_memory = False)
    
    return trainLoader, testLoader