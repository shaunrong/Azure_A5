# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check_extra.py 1d
    sanity_check_extra.py 1e
"""
import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.utils
from docopt import docopt

from char_decoder import CharDecoder
from nmt_model import NMT
from utils import pad_sents_char
from vocab import Vocab, VocabEntry

import torch.nn as nn
import numpy as np

import unittest

from cnn import CNN
from highway import Highway


BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0
NUM_FILTER = 4
KERNEl_SIZE = 3
MAX_WORD_LEN  = 8

def reinitialize_layers(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)
    with torch.no_grad():
        model.apply(init_weights)


class HighwaySanityChecks(unittest.TestCase):

    def test_shape(self):
        print("-" * 80)
        print("Running Sanity Check for Question 1d: Highway Shape")
        print("-" * 80)
        batch_size, word_embed_size = 64, 40
        highway = Highway(word_embed_size)

        x_conv_out = torch.randn([batch_size, word_embed_size])
        x_word_emb = highway.forward(x_conv_out)

        self.assertEqual(x_word_emb.shape, (batch_size, word_embed_size))
        self.assertEqual(x_word_emb.shape, x_conv_out.shape)
        print("Sanity Check Passed for Question 1d: Highway Shape!")
        print("-" * 80)

    def test_gate_bypass(self):
        print("-" * 80)
        print("Running Sanity Check for Question 1d: Highway Bypass")
        print("-" * 80)

        batch_size, word_embed_size = 64, 40
        highway = Highway(word_embed_size)
        highway.gate.weight.data[:, :] = 0.0
        highway.gate.bias.data[:] = - float('inf')

        x_conv_out = torch.randn([batch_size, word_embed_size])
        x_word_emb = highway.forward(x_conv_out)

        self.assertTrue(torch.allclose(x_conv_out, x_word_emb))
        print("Sanity Check Passed for Question 1d: Highway Bypass!")
        print("-" * 80)

    def test_gate_projection(self):
        print("-" * 80)
        print("Running Sanity Check for Question 1d: Highway Gate")
        print("-" * 80)
        batch_size, word_embed_size = 64, 40
        highway = Highway(word_embed_size)
        highway.projection.weight.data = torch.eye(word_embed_size)
        highway.projection.bias.data[:] = 0.0
        highway.gate.weight.data[:, :] = 0.0
        highway.gate.bias.data[:] = float('inf')

        x_conv_out = torch.rand([batch_size, word_embed_size])
        x_word_emb = highway(x_conv_out)

        self.assertTrue(torch.allclose(x_conv_out, x_word_emb))
        print("Sanity Check Passed for Question 1d: Highway Gate!")
        print("-" * 80)

    def reinitialize_layers(self, model):
        """ Reinitialize the Layer Weights for Sanity Checks.
        """

        def init_weights(m):
            if type(m) == nn.Linear:
                m.weight.data.fill_(0.3)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif type(m) == nn.Embedding:
                m.weight.data.fill_(0.15)
            elif type(m) == nn.Dropout:
                nn.Dropout(DROPOUT_RATE)

        with torch.no_grad():
            model.apply(init_weights)

    def highway_generate_data(self):
        """
        Unit test data generator for Highway
        """
        conv_input = np.random.rand(BATCH_SIZE, EMBED_SIZE)
        W_proj = np.ones((EMBED_SIZE, EMBED_SIZE)) * 0.3
        b_proj = np.ones(EMBED_SIZE) * 0.1

        W_gate = np.ones((EMBED_SIZE, EMBED_SIZE)) * 0.3
        b_gate = np.ones(EMBED_SIZE) * 0.1

        def relu(inpt):
            return np.maximum(inpt, 0)

        def sigmoid(inpt):
            return 1. / (1 + np.exp(-inpt))

        x_proj = relu(conv_input.dot(W_proj) + b_proj)
        x_gate = sigmoid(conv_input.dot(W_gate) + b_gate)
        x_highway = x_gate * x_proj + (1 - x_gate) * conv_input

        np.save('sanity_check_handmade_data/highway_conv_input.npy', conv_input)
        np.save('sanity_check_handmade_data/highway_output.npy', x_highway)


    def highway_data_check(self, highway):
        """
        Sanity check for highway.py, basic shape check and forward pass check
        """
        print("-" * 80)
        print("Running Sanity Check for Question 1d: Highway Data")
        print("-" * 80)

        BATCH_SIZE = 5
        EMBED_SIZE = 3
        HIDDEN_SIZE = 3
        DROPOUT_RATE = 0.0
        NUM_FILTER = 4
        KERNEl_SIZE = 3
        MAX_WORD_LEN = 8

        reinitialize_layers(highway)

        highway_generate_data()

        inpt = torch.from_numpy(np.load('sanity_check_handmade_data/highway_conv_input.npy').astype(np.float32))
        outp_expected = torch.from_numpy(np.load('sanity_check_handmade_data/highway_output.npy').astype(np.float32))

        with torch.no_grad():
            outp = highway(inpt)

        outp_expected_size = (BATCH_SIZE, EMBED_SIZE)
        assert (outp.numpy().shape == outp_expected_size), \
            "Highway output shape is incorrect it should be:\n{} but is:\n{}".format(outp.numpy().shape,
                                                                                     outp_expected_size)
        assert (np.allclose(outp.numpy(), outp_expected.numpy())), \
            "Highway output is incorrect: it should be:\n {} but is:\n{}".format(outp_expected, outp)
        print("Sanity Check Passed for Question 1d: Highway data!")
        print("-" * 80)


class CNNSanityChecks(unittest.TestCase):

    def test_shape(self):
        print("-" * 80)
        print("Running Sanity Check for Question 1e: CNN Shape")
        print("-" * 80)
        max_word_len = 15
        batch_size, char_embed_size, num_filters, window_size = 64, 20, 80, 5
        cnn = CNN(char_embed_size, num_filters, max_word_len, window_size)

        x_emb = torch.randn([batch_size, char_embed_size, max_word_len])
        x_conv_out = cnn.forward(x_emb)

        self.assertEqual(x_conv_out.shape, (batch_size, num_filters))
        print("Sanity Check Passed for Question 1e: CNN Shape!")
        print("-" * 80)


def main():
    """ Main func.
    """

    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert (
            torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0 or greater".format(
        torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    if args['1d']:
        highway = HighwaySanityChecks()
        highway.test_shape()
        highway.test_gate_bypass()
        highway.test_gate_projection()
        # highway.highway_data_check(highway)
    elif args['1e']:
        cnn = CNNSanityChecks()
        cnn.test_shape()
    else:
        raise RuntimeError('invalid run mode')

if __name__ == '__main__':
    main()