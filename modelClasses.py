import torch as t
import torch.nn as nn
import numpy as np


class Model(nn.Module):

    def __init__(self, inSize, outSize, hidSize):
        super().__init__()
        self.linLay1 = nn.Linear(in_features=inSize, out_features=round(12.8 * hidSize))
        self.linLay2 = nn.Linear(in_features=round(12.8 * hidSize), out_features=outSize)
        self.smallLinLay1 = nn.Linear(in_features=inSize, out_features=round(0.64 * hidSize))
        self.smallLinLay2 = nn.Linear(in_features=round(0.64 * hidSize), out_features=round(0.64 * hidSize))
        self.smallLinLay3 = nn.Linear(in_features=round(0.64 * hidSize), out_features=round(1.28 * hidSize))
        self.smallLinLay4 = nn.Linear(in_features=round(1.28 * hidSize), out_features=outSize)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.sm = nn.Softmax()


    def cont2contLin(self, inputs):
        process = self.linLay1(inputs)
        process = self.relu(process)
        outputs = self.linLay2(process)
        return outputs

    def cont2contNonLin(self, inputs):
        process = self.linLay1(inputs)
        process = self.sig(process)
        outputs = self.linLay2(process)
        return outputs

    def cont2binLin(self, inputs):
        process = self.linLay1(inputs)
        process = self.relu(process)
        process = self.linLay2(process)
        outputs = self.sig(process)
        return outputs


    def cont2binNonLin(self, inputs):
        process = self.linLay1(inputs)
        process = self.sig(process)
        process = self.linLay2(process)
        outputs = self.sig(process)
        return outputs

    def cont2binNonLin2(self, inputs):
        process = self.smallLinLay1(inputs)
        process = self.relu(process)
        process = self.smallLinLay2(process)
        process = self.relu(process)
        process = self.smallLinLay3(process)
        process = self.relu(process)
        process = self.smallLinLay4(process)
        outputs = self.sig(process)
        return outputs
