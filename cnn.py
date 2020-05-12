#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    '''
    Class of Convolutional network
    '''
    def __init__(self, char_embed_size, num_filters, m_word=21, k=5):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=char_embed_size,
                              out_channels=num_filters,
                              kernel_size=k)
        self.max_pool = nn.MaxPool1d(kernel_size=m_word - k + 1)
        # print('CNN parameter:', char_embed_size, num_filters, m_word, k)
        # print('kernel size for maxpool:', m_word - k + 1)

    def forward(self, X_reshaped):
        # print('IN CNN, shape of X_reshaped:', X_reshaped.size())
        X_conv = self.conv(X_reshaped)
        # print('IN CNN, shape of X_conv:', X_conv.size())
        X_relu = F.relu(X_conv)
        # print('IN CNN, shape of X_relu:', X_relu.size())
        X_maxpool = self.max_pool(X_relu)

        # print('IN CNN, shape of X_maxpool:', X_maxpool.size())
        X_conv_out = torch.squeeze(X_maxpool, -1)
        # print('IN CNN, shape of X_conv_out:', X_conv_out.size())
        return X_conv_out


### END YOUR CODE

