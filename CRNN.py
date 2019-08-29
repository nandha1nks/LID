'''
Building the basic CRNN network
Input: Spectrogram Image ( dim )
Output: Probabilities ( 6 x 1 )

Call Model for training as:

crnn = CRNN(inshape,outClasses,rnnHiddenSize,rnnLayers)
crnn = crrn.cuda()
loss = nn.NLLLoss()  #If we output without softmax in model, then CrossEntropyLoss. Can softmax later for probabilites.
optimizer = optim.Adam(crnn.parameters(),lr=0.01) #Need to Tune

'''

import torch
import torch.nn as nn
import torch.optim as optim

class CRNN(nn.Module):
    def __init__(self, inShape = (3,129,500), outClasses=5, rnnHiddenSize = 256, rnnLayers=2):
        
        # inshape = (vertical/FreqAxis,horiz/TimeAxis,channels)
        
        super(CRNN, self).__init__()
        #assert imgH % 16 == 0, 'imgH has to be a multiple of 16' WHY??

        kerSize = [3, 3, 3, 3, 3, 3, 3]
        pad = [1, 1, 1, 1, 1, 1, 0]
        stride = [1, 1, 1, 1, 1, 1, 1]
        filters = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()

        def convRelu(i, batchNorm=True, leakyRelu = False, maxPool = True):
            channelPrev = inShape[0] if i == 0 else filters[i - 1]
            channelNext = filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(channelPrev, channelNext,
                                     kerSize[i], stride[i], pad[i]))
            if batchNorm:
                cnn.add_module('batchNorm{0}'.format(i),
                               nn.BatchNorm2d(channelNext))
            if leakyRelu:
                cnn.add_module('leakyRelu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
            
            if maxPool:
                cnn.add_module('pooling{0}'.format(i), nn.MaxPool2d(2, 2))
        
        convRelu(0)
        convRelu(1)
        convRelu(2, maxPool = False)
        convRelu(3)
        convRelu(4, maxPool = False)
        convRelu(5)
        convRelu(6)

        self.cnn = cnn
        self.rnn = nn.LSTM(filters[-1],rnnHiddenSize,
                        rnnLayers, batch_first = True, bidirectional=True)
        
        self.fc = nn.Linear(rnnHiddenSize*rnnLayers,outClasses)

    def forward(self, input):
        
        featureSet = self.cnn(input)
        batchSize, channels, height, timeSeq = featureSet.size()
        
        assert height == 1 # Need to verify if the height turns out to be 1 IMP
        featureSet = featureSet.squeeze(2)  # Remove height
        
        featureSet = featureSet.permute(2, 0, 1)  # [timeSeq, batchSize, channels]
        out = self.rnn(featureSet) # [timeSeq, batchSize, rnnLayers*channels]
        out = out.permute(1, 0, 2) # [batchSize, timeSeq, rnnLayers*channels]
        
        prob = nn.LogSoftmax(self.fc(out))
        return prob