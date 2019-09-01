'''
Basic CRNN network
Input: Spectrogram Image ( dim )
Output: Probabilities ( 6 x 1 )

Call Model for training as:

crnn = CRNN(inshape,outClasses,rnnHiddenSize,rnnLayers)
crnn = crrn.cuda()
loss = nn.NLLLoss()  #If we output without softmax in model, then CrossEntropyLoss. Can softmax later for probabilites.
optimizer = optim.Adam(crnn.parameters(),lr=0.01) #Need to Tune
'''
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, config):
        
        # inshape = (vertical/FreqAxis,horiz/TimeAxis,channels)
        
        super(CRNN, self).__init__()
        #assert imgH % 16 == 0, 'imgH has to be a multiple of 16' WHY??

        kerSize = [3, 3, 3, 3, 3, 3, 3]
        pad = [0, 0, 0, 0, 0, 0, 0]
        stride = [1, 1, 1, 1, 1, 1, 1]
        filters = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()

        def convRelu(i, batchNorm=True, maxPool = True):
            channelPrev = config['inShape'][0] if i == 0 else filters[i - 1]
            channelNext = filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(channelPrev, channelNext,
                                     kerSize[i], stride[i], pad[i]))
            if batchNorm:
                cnn.add_module('batchNorm{0}'.format(i),
                               nn.BatchNorm2d(channelNext))
            if config['cnnActivation'] == 'leaky_relu':
                cnn.add_module('leakyRelu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            elif config['cnnActivation'] == 'relu':
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
            elif config['cnnActivation'] == 'relu':
                cnn.add_module('relu{0}'.format(i), nn.ELU(1.0,True))
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
        self.rnn = nn.LSTM(filters[-1],config['rnnHiddenSize'],
                        config['rnnLayers'], batch_first = True, 
                        bidirectional=True)
        
        self.fc = nn.Linear(config['fcHiddenSize']*config['fcLayers'],
                            config['outClasses'])
        self.output = nn.LogSoftmax(0)
        
    def forward(self, input):
        
        featureSet = self.cnn(input)
        batchSize, channels, height, timeSeq = featureSet.size()
        #print(featureSet.size())
        assert height == 1 # Need to verify if the height turns out to be 1 IMP
        featureSet = featureSet.squeeze(2)  # Remove height
        
        featureSet = featureSet.permute(2, 0, 1)  # [timeSeq, batchSize, channels]
        out, _ = self.rnn(featureSet) # [timeSeq, batchSize, 2*rnnHiddenSize]
        
        out = out[timeSeq-1,:,:] # [1, batchSize, 2*rnnHiddenSize]
        # Taking only last time step output from RNN (as is in Literature)
        out = out.squeeze(0) # [batchSize, 2*rnnHiddenSize
        for i in range(config['fcLayers']-1):
            out = self.fc(out)
        prob = self.output(self.fc(out))
        return prob 