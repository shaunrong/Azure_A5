#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    '''
    Class of the highway network
    '''

    def __init__(self, e_word):
        ''' Init highway network model
        @param e_word (int): the size of the final word embedding for word

        Initialization:
        self.projection. Linear layer with bias. Called W{proj} in PDF.
        self.gate. Linear Layer with bias. Called W{gate} in PDF.
        '''
        super(Highway, self).__init__()
        self.projection = nn.Linear(e_word, e_word, bias=True)
        self.gate = nn.Linear(e_word, e_word, bias=True)


    def forward(self, X_conv_out):
        X_proj = F.relu(self.projection(X_conv_out))
        X_gate = torch.sigmoid(self.gate(X_conv_out))
        X_highway = torch.add(torch.mul(X_proj, X_gate), torch.mul((1 - X_gate), X_conv_out))
        return X_highway

# class HighwaySanityCheck():


### END YOUR CODE 

