# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/3/12 下午2:31
@summary:
"""
import torch
import jieba
import argparse
from codecs import open
import torch.nn as nn
from models.pytorch_rnn import RNNModel, LSTMModel, GRUModel, BiLSTMModel

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=128, help="Total examples' number of a batch for training.")
parser.add_argument("--max_len", type=int, default=128, help="The text length")
parser.add_argument('--network', choices=['lstm', 'bilstm', 'gru', 'rnn'],
    default="bilstm", help="Select which network to train, defaults to bilstm.")

args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
def build_vocab(make_vocab=True, vocab_path=""):
    """读取数据"""
    vocab_lists = set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                vocab = line.strip('\n')
                vocab_lists.add(vocab)

    vocab_list = list(vocab_lists)
    # 如果make_vocab为True，还需要返回vocab2id
    if make_vocab:
        vocab2id = dict(zip(vocab_list, range(len(vocab_lists))))
        return vocab_lists, vocab2id
    else:
        return vocab_lists

def sort_by_lengths(corpus_lists, label_lists, lengths_list):
    pairs = list(zip(range(len(lengths_list)), lengths_list))
    indices = sorted(range(len(pairs)), key=lambda k: pairs[k][0],  reverse=True)

    corpus_lists_ = [corpus_lists[ind] for ind in indices]
    label_lists_ = [label_lists[ind] for ind in indices]
    lengths_list_ = [lengths_list[ind] for ind in indices]

    return corpus_lists_, label_lists_, lengths_list_

def read_data(vocab2id, max_len, path="", flag='train'):
    """读取数据"""
    corpus_lists = []
    label_lists = []
    lengths_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            items = line.strip().split("\t")
            if flag == 'train':
                assert len(items) == 2
                label_lists.append(int(items[0]))
                text_list = [vocab2id.get(vocab, 0) for vocab in jieba.cut(items[1])]
                lengths_list.append(len(text_list))
                if len(text_list) < max_len:
                    text_list = text_list + [vocab2id['[PAD]'] for i in range(max_len - len(text_list))]

                corpus_lists.append(text_list[:max_len])
            if flag == 'dev':
                assert len(items) == 3
                label_lists.append(int(items[1]))
                text_list = [vocab2id.get(vocab, 0) for vocab in jieba.cut(items[2])]
                lengths_list.append(len(text_list))
                if len(text_list) < max_len:
                    text_list = text_list + [vocab2id['[PAD]']  for i in range(max_len - len(text_list))]
                corpus_lists.append(text_list[:max_len])

    return corpus_lists, label_lists, lengths_list

def bulid_model(vocab_size, num_classes):
    """"""
    # Constructs the network.
    network = args.network.lower()
    if network == 'rnn':
        model = RNNModel(vocab_size, num_classes)
    elif network == 'lstm':
        model = LSTMModel(vocab_size, num_classes)
    elif network == 'gru':
        model = GRUModel(vocab_size, num_classes)
    elif network == 'bilstm':
        model = LSTMModel(vocab_size, num_classes)
    else:
        raise ValueError("Unknown network: %s, it must be one of lstm, bilstm, gru, rnn" % network)
    return model

def validate(model, loss_func, dev_corpus_lists, dev_label_lists, dev_lengths_list):
    """validate"""
    _best_val_loss = None
    B = args.batch_size
    with torch.no_grad():
        val_losses = 0.
        val_step = 0
        for ind in range(0, len(dev_corpus_lists), args.epochs+1):
            val_step += 1
            # 准备batch数据
            batch_sents = train_corpus_lists[ind:ind + B]
            batch_labels = train_label_lists[ind:ind + B]
            batch_lengths = train_lengths_list[ind:ind + B]
            batch_sents, batch_labels, batch_lengths = sort_by_lengths(batch_sents, batch_labels, batch_lengths)

            batch_sents = torch.tensor(batch_sents, dtype=torch.int64)
            batch_labels = torch.tensor(batch_labels, dtype=torch.int64)

            # forward
            scores = model(batch_sents, batch_lengths)

            # 计算损失
            loss = loss_func(scores, batch_labels)
            val_losses += loss
        val_loss = val_losses / val_step

        # if val_loss < _best_val_loss:
        #     print("保存模型...")

        return val_loss

#
def train(vocab_size, num_classes, train_corpus_lists, train_label_lists, train_lengths_list,
          dev_corpus_lists, dev_label_lists, dev_lengths_list):
    model = bulid_model(vocab_size, num_classes)
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    # 初始化其他指标
    print_step = 100
    B = args.batch_size
    for e in range(1, args.epochs+1):
        step = 0
        losses = 0.
        for ind in range(0, len(train_corpus_lists), B):

            batch_sents = train_corpus_lists[ind:ind+B]
            batch_labels = train_label_lists[ind:ind+B]
            batch_lengths = train_lengths_list[ind:ind + B]
            batch_sents, batch_labels, batch_lengths = sort_by_lengths(batch_sents, batch_labels, batch_lengths)

            batch_sents = torch.tensor(batch_sents, dtype=torch.int64)
            batch_labels = torch.tensor(batch_labels, dtype=torch.int64)
            print(batch_sents)
            print(batch_sents.shape)

            # forward
            scores = model(batch_sents, batch_lengths)
            optimizer.zero_grad()

            loss = loss_func(scores, batch_labels)
            losses += loss

            loss.backward()
            optimizer.step()
            step += 1

            if step % print_step == 0:
                total_step = (len(train_corpus_lists) // B + 1)
                print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                    e, step, total_step,
                    100. * step / total_step,
                    losses / print_step
                ))
                losses = 0.

        # 每轮结束测试在验证集上的性能，保存最好的一个

        val_loss = validate(model, loss_func, dev_corpus_lists, dev_label_lists, dev_lengths_list)
        print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

if __name__ == '__main__':
    vocab_path = '/home/mesie/python/data/nlp/senta_word_dict.txt'
    train_path = "/home/mesie/python/data/nlp/ChnSentiCorp/train.tsv"
    dev_path = "/home/mesie/python/data/nlp/ChnSentiCorp/dev.tsv"
    label_list = ["0", "1"]

    vocab_lists, vocab2id = build_vocab(vocab_path=vocab_path)
    train_corpus_lists, train_label_lists, train_lengths_list = read_data(vocab2id, max_len=args.max_len, path=train_path)
    dev_corpus_lists, dev_label_lists, dev_lengths_list = read_data(vocab2id, max_len=args.max_len, path=dev_path, flag='dev')

    vocab_size = len(vocab_lists)
    num_classes = len(label_list)

    train(vocab_size, num_classes, train_corpus_lists, train_label_lists, train_lengths_list,
          dev_corpus_lists, dev_label_lists, dev_lengths_list)
    # print(torch.Tensor(label_lists))
