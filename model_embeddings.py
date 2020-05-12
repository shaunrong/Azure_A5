#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code
        self.vocab = vocab
        self.embed_size = embed_size
        self.embed_char = 50
        self.dropout_rate = 0.3
        self.padding_idx = vocab.char2id['<pad>']

        self.char_embeddings = nn.Embedding(len(vocab.char2id),
                                            self.embed_char,
                                            self.padding_idx)
        self.CNN = CNN(self.embed_char, embed_size)
        self.Highway = Highway(embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        ### YOUR CODE HERE for part 1f

        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        char_embeddings = self.char_embeddings(input_tensor)
        sent_len, batch_size, max_word, char_embed_size = char_embeddings.shape
        # Reshape char embedding
        X_reshaped = char_embeddings.view(sent_len * batch_size, max_word, char_embed_size).transpose(1, 2)
        X_conv_out  = self.CNN(X_reshaped)
        X_highway = self.Highway(X_conv_out)
        output = self.dropout(X_highway)
        output = output.view(sent_len, batch_size, self.embed_size)
        return output
        ### END YOUR CODE
