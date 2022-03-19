# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/3/5 下午4:51
@summary:
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNNModel(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_size=128, hidden_size=128):
        """
        :param vocab_size: 字典的大小
        :param num_classes: 分类
        :param emb_size:  词向量的维数
        :param hidden_size: 隐向量的维数
        """
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)

        self.lin = nn.Linear(2*hidden_size, num_classes)

    def forward(self, text):
        emb = self.embedding(text)  # [B, L, emb_size]

        rnn_out, _ = self.rnn(emb)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out = rnn_out[:, -1, :]

        out = self.lin(rnn_out)  # [B, out_size]

        return out

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_size=128, hidden_size=128):
        """
        :param vocab_size: 字典的大小
        :param num_classes: 分类
        :param emb_size:  词向量的维数
        :param hidden_size: 隐向量的维数
        """
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=False)

        self.lin = nn.Linear(2*hidden_size, num_classes)

    def forward(self, text):
        emb = self.embedding(text)  # [B, L, emb_size]
        rnn_out, _ = self.lstm(emb)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out = rnn_out[:, -1, :]

        out = self.lin(rnn_out)  # [B, out_size]

        return out

class GRUModel(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_size=128, hidden_size=128):
        """
        :param vocab_size: 字典的大小
        :param num_classes: 分类
        :param emb_size:  词向量的维数
        :param hidden_size: 隐向量的维数
        """
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)

        self.lin = nn.Linear(2*hidden_size, num_classes)

    def forward(self, text):
        emb = self.embedding(text)  # [B, L, emb_size]

        rnn_out, _ = self.gru(emb)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out = rnn_out[:, -1, :]

        out = self.lin(rnn_out)  # [B, out_size]

        return out

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_size=128, hidden_size=128):
        """
        :param vocab_size: 字典的大小
        :param num_classes: 分类
        :param emb_size:  词向量的维数
        :param hidden_size: 隐向量的维数
        """
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.lin = nn.Linear(2*hidden_size, num_classes)

    def forward(self, text):
        emb = self.embedding(text)  # [B, L, emb_size]

        rnn_out, _ = self.bilstm(emb)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out = rnn_out[:, -1, :]
        out = self.lin(rnn_out)  # [B, out_size]

        return out