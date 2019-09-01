import random
import pickle

def createCombos():
    fileName = 'combos'
    epochs = [20,40,60,80,100,120,140]
    optimizers = ['Adam', 'Momentum','Adagrad','RMSProp']
    learningRates = [0.001,0.003,0.01,0.03,0.1,0.3,1,2]
    batchSizes = [32,64,128,256,512]
    rnnHiddenSizes = [128,256,512]
    rnnLayers = [1,2,4]
    noFCLayers = [1,2,4]
    fcSizes = [128,256,512]
    cnnActivations = ['relu','elu','leaky_relu']
    
    lists = []
    for epo in epochs:
        for opt in optimizers:
            for lr in learningRates:
                for bS in batchSizes:
                    for rnnHS in rnnHiddenSizes:
                        for rnnL in rnnLayers:
                            for noF in noFCLayers:
                                for fcSize in fcSizes:
                                    for cnnAct in cnnActivations:
                                        lists.append({
                                                'iterations':epo,
                                                'optimizer': opt,
                                                'learningRate':lr,
                                                'batchSize': bS,
                                                'rnnHiddenSize': rnnHS,
                                                'rnnLayers': rnnL,
                                                'fcHiddenSize':fcSize,
                                                'fcLayers': noF,
                                                'cnnActivation': cnnAct
                                                })
    
    random.shuffle(lists)
    file = open(fileName,'wb+')
    pickle.dump(lists[:70],file)
    file.close()
    return fileName