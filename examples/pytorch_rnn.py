# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/3/12 下午2:31
@summary:
"""
import torch
import jieba
import random
import argparse
from codecs import open
import torch.nn as nn
from sklearn import metrics
from models.pytorch_rnn import RNNModel, LSTMModel, GRUModel, BiLSTMModel

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=30, help="Number of epoches for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
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
    indices = [ind[0] for ind in sorted(pairs, key=lambda k: k[1],  reverse=True)]
    corpus_lists_ = [corpus_lists[ind] for ind in indices]
    label_lists_ = [label_lists[ind] for ind in indices]
    lengths_list_ = [lengths_list[ind] for ind in indices]

    return corpus_lists_, label_lists_, lengths_list_

def sample_data(corpus_lists, label_lists):
    """打乱数据"""
    pairs = list(zip(range(len(label_lists)), label_lists))
    random.shuffle(pairs)
    indices = [pair[0] for pair in pairs]

    corpus_lists_ = [corpus_lists[ind] for ind in indices]
    label_lists_ = [pair[1] for pair in pairs]

    return corpus_lists_, label_lists_

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

def read_test_data(texts, vocab2id, max_len):
    """测试数据"""
    corpus_lists = []
    for text in texts:
        text_list = [vocab2id.get(vocab, 0) for vocab in jieba.cut(text)]
        if len(text_list) < max_len:
            text_list = text_list + [vocab2id['[PAD]'] for i in range(max_len - len(text_list))]

        corpus_lists.append(text_list[:max_len])

    return corpus_lists


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
        model = BiLSTMModel(vocab_size, num_classes)
    else:
        raise ValueError("Unknown network: %s, it must be one of lstm, bilstm, gru, rnn" % network)
    return model

def validate(model, loss_func, dev_corpus_lists, dev_label_lists, dev_lengths_list):
    """validate"""
    _best_val_loss = 0.4
    _best_val_acc = 0.87
    flag = False
    B = args.batch_size
    with torch.no_grad():
        val_losses = 0.
        val_step = 0
        val_accs = 0
        for ind in range(0, len(dev_corpus_lists), args.epochs+1):
            val_step += 1
            # 准备batch数据
            batch_sents = dev_corpus_lists[ind:ind + B]
            batch_labels = dev_label_lists[ind:ind + B]
            batch_lengths = dev_lengths_list[ind:ind + B]
            # batch_sents, batch_labels, batch_lengths = sort_by_lengths(batch_sents, batch_labels, batch_lengths)

            batch_sents = torch.tensor(batch_sents, dtype=torch.int64)
            batch_labels = torch.tensor(batch_labels, dtype=torch.int64)

            # forward
            scores = model(batch_sents)
            predic = torch.max(scores.data, 1)[1].cpu()
            acc = metrics.accuracy_score(batch_labels.data.cpu(), predic)
            # 计算损失
            loss = loss_func(scores, batch_labels)
            val_losses += loss
            val_accs += acc
        val_loss = val_losses / val_step
        val_acc = val_accs / val_step
        if val_loss < _best_val_loss or val_acc > _best_val_acc:
            save_path = f"{args.save_dir}{args.network}.ckpt"
            torch.save(model.state_dict(), save_path)
            flag = True

        return val_loss, val_acc, flag

#
def train(vocab_size, num_classes, train_corpus_lists, train_label_lists, train_lengths_list,
          dev_corpus_lists, dev_label_lists, dev_lengths_list):
    model = bulid_model(vocab_size, num_classes)
    model.train()
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    # 初始化其他指标
    print_step = 10
    B = args.batch_size
    for e in range(1, args.epochs+1):
        step = 0
        losses = 0.
        # train_corpus_lists, train_label_lists = sample_data(train_corpus_lists, train_label_lists)
        for ind in range(0, len(train_corpus_lists), B):
            step += 1
            batch_sents = train_corpus_lists[ind:ind+B]
            batch_labels = train_label_lists[ind:ind+B]
            batch_lengths = train_lengths_list[ind:ind + B]
            # batch_sents, batch_labels, batch_lengths = sort_by_lengths(batch_sents, batch_labels, batch_lengths)

            batch_sents = torch.tensor(batch_sents, dtype=torch.int64).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.int64).to(device)

            # forward
            scores = model(batch_sents)
            predic = torch.max(scores.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(batch_labels.data.cpu(), predic)

            model.zero_grad()

            loss = loss_func(scores, batch_labels).to(device)
            losses += loss

            loss.backward()
            optimizer.step()

            if step % print_step == 0:
                total_step = (len(train_corpus_lists) // B + 1)
                print("Epoch {}, step/total_step: {}/{} Acc:{:.4f} Loss:{:.4f}".format(
                    e, step, total_step,  train_acc, losses / step))

        # 每轮结束测试在验证集上的性能，保存最好的一个
        val_loss, val_acc, flag = validate(model, loss_func, dev_corpus_lists, dev_label_lists, dev_lengths_list)
        print("Epoch {}, Val Acc:{:.4f}, Val Loss:{:.4f}".format(e, val_acc, val_loss))
        if flag:
            break

    # # 保存模型
    # save_path = f"{args.save_dir}{args.network}.ckpt"
    # torch.save(model.state_dict(), save_path)

def predict(vocab_size, num_classes, vocab2id):
    """模型预测"""
    model = bulid_model(vocab_size, num_classes)
    save_path = f"{args.save_dir}{args.network}.ckpt"
    model.load_state_dict(torch.load(save_path))
    model.eval()
    texts = [
        '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般',
        '东西不错，不过有人不太喜欢镜面的，我个人比较喜欢，总之还算满意。',
        '位置不很方便，周围乱哄哄的，卫生条件也不如其他如家的店。以后绝不会再住在这里。'
    ]
    max_len = args.max_len
    corpus_lists = read_test_data(texts, vocab2id, max_len)
    batch_sents = torch.tensor(corpus_lists, dtype=torch.int64)
    scores = model(batch_sents)
    print(scores.data)
    predic = torch.max(scores.data, 1)[1].numpy()
    print(predic)


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
    # predict(vocab_size, num_classes, vocab2id)
    # print(torch.Tensor(label_lists))
