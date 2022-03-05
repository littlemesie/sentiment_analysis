# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/2/26 下午3:58
@summary:
"""
from functools import partial
import argparse
import os
import random

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import load_dataset

from models.paddle_rnn import BoWModel, BiLSTMAttentionModel, CNNModel, LSTMModel, GRUModel, RNNModel, \
    SelfInteractiveAttention

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument('--network', choices=['bow', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'],
    default="bilstm", help="Select which network to train, defaults to bilstm.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams', help="The path of model parameter to be loaded.")

args = parser.parse_args()
# yapf: enable

def set_seed(seed=1000):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def convert_example(example, tokenizer, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks.
    It use `jieba.cut` to tokenize text.

    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj: paddlenlp.data.JiebaTokenizer): It use jieba to cut the chinese string.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        valid_length(obj:`int`): The input sequence valid length.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """

    input_ids = tokenizer.encode(example["text"])
    valid_length = np.array(len(input_ids), dtype='int64')
    input_ids = np.array(input_ids, dtype='int64')

    if not is_test:
        label = np.array(example["label"], dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader

def read_train(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            items = line.strip().split("\t")
            assert len(items) == 2
            example = {
                "label": int(items[0]),
                "text": items[1]
            }

            yield example

def read_dev(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            items = line.strip().split("\t")
            assert len(items) == 3
            example = {
                "label": int(items[1]),
                "text": items[2]
            }

            yield example

def preprocess_prediction_data(data, tokenizer):
    """
    It process the prediction data as the format used as training.

    Args:
        data (obj:`List[str]`): The prediction data whose each element is  a tokenized text.
        tokenizer(obj: paddlenlp.data.JiebaTokenizer): It use jieba to cut the chinese string.

    Returns:
        examples (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).

    """
    examples = []
    for text in data:
        ids = tokenizer.encode(text)
        examples.append([ids, len(ids)])
    return examples

def bulid_model(vocab_size, num_classes, pad_token_id):
    """"""
    # Constructs the network.
    network = args.network.lower()
    if network == 'bow':
        model = BoWModel(vocab_size, num_classes, padding_idx=pad_token_id)
    elif network == 'bigru':
        model = GRUModel(
            vocab_size,
            num_classes,
            direction='bidirect',
            padding_idx=pad_token_id)
    elif network == 'bilstm':
        model = LSTMModel(
            vocab_size,
            num_classes,
            direction='bidirect',
            padding_idx=pad_token_id)
    elif network == 'bilstm_attn':
        lstm_hidden_size = 196
        attention = SelfInteractiveAttention(hidden_size=2 * lstm_hidden_size)
        model = BiLSTMAttentionModel(
            attention_layer=attention,
            vocab_size=vocab_size,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=num_classes,
            padding_idx=pad_token_id)
    elif network == 'birnn':
        model = RNNModel(
            vocab_size,
            num_classes,
            direction='bidirect',
            padding_idx=pad_token_id)
    elif network == 'cnn':
        model = CNNModel(vocab_size, num_classes, padding_idx=pad_token_id)
    elif network == 'gru':
        model = GRUModel(
            vocab_size,
            num_classes,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
    elif network == 'lstm':
        model = LSTMModel(
            vocab_size,
            num_classes,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
    elif network == 'rnn':
        model = RNNModel(
            vocab_size,
            num_classes,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
    else:
        raise ValueError(
            "Unknown network: %s, it must be one of bow, lstm, bilstm, cnn, gru, bigru, rnn, birnn and bilstm_attn."
            % network)
    model = paddle.Model(model)

    return model

def train(vocab_path, train_path, dev_path):
    """训练模型"""
    # Load vocab
    vocab = Vocab.load_vocabulary(
        vocab_path, unk_token='[UNK]', pad_token='[PAD]')
    # Loads dataset.
    # train_ds, dev_ds = load_dataset("chnsenticorp", splits=["train", "dev"])
    train_ds = load_dataset(read_train, data_path=train_path, lazy=False)
    dev_ds = load_dataset(read_dev, data_path=dev_path, lazy=False)
    for example in train_ds[0:2]:
        print(example)

    vocab_size = len(vocab)
    num_classes = len(label_list)
    pad_token_id = vocab.to_indices('[PAD]')

    model = bulid_model(vocab_size, num_classes, pad_token_id)

    # Reads data and generates mini-batches.
    tokenizer = JiebaTokenizer(vocab)
    trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    train_loader = create_dataloader(
        train_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='train',
        batchify_fn=batchify_fn)
    dev_loader = create_dataloader(
        dev_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='validation',
        batchify_fn=batchify_fn)

    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=args.lr)

    # Defines loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    model.prepare(optimizer, criterion, metric)

    # Loads pre-trained parameters.
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    # Starts training and evaluating.
    callback = paddle.callbacks.ProgBarLogger(log_freq=20, verbose=3)
    model.fit(train_loader,
              dev_loader,
              epochs=args.epochs,
              save_dir=args.save_dir,
              save_freq=5,
              callbacks=callback)

def predict(vocab_path, label_list, batch_size=1, pad_token_id=0):
    """预测"""
    # Load vocab
    vocab = Vocab.load_vocabulary(
        vocab_path, unk_token='[UNK]', pad_token='[PAD]')
    # Loads dataset.
    label_map = {0: 'negative', 1: 'positive'}

    vocab_size = len(vocab)
    num_classes = len(label_list)
    pad_token_id = vocab.to_indices('[PAD]')

    network = args.network.lower()
    if network == 'bow':
        model = BoWModel(vocab_size, num_classes, padding_idx=pad_token_id)
    elif network == 'bigru':
        model = GRUModel(
            vocab_size,
            num_classes,
            direction='bidirect',
            padding_idx=pad_token_id)
    elif network == 'bilstm':
        model = LSTMModel(
            vocab_size,
            num_classes,
            direction='bidirect',
            padding_idx=pad_token_id)
    elif network == 'bilstm_attn':
        lstm_hidden_size = 196
        attention = SelfInteractiveAttention(hidden_size=2 * lstm_hidden_size)
        model = BiLSTMAttentionModel(
            attention_layer=attention,
            vocab_size=vocab_size,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=num_classes,
            padding_idx=pad_token_id)
    elif network == 'birnn':
        model = RNNModel(
            vocab_size,
            num_classes,
            direction='bidirect',
            padding_idx=pad_token_id)
    elif network == 'cnn':
        model = CNNModel(vocab_size, num_classes, padding_idx=pad_token_id)
    elif network == 'gru':
        model = GRUModel(
            vocab_size,
            num_classes,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
    elif network == 'lstm':
        model = LSTMModel(
            vocab_size,
            num_classes,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
    elif network == 'rnn':
        model = RNNModel(
            vocab_size,
            num_classes,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
    else:
        raise ValueError(
            "Unknown network: %s, it must be one of bow, lstm, bilstm, cnn, gru, bigru, rnn, birnn and bilstm_attn."
            % network)

    # Loads model parameters.
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % args.params_path)

    # Firstly pre-processing prediction data  and then do predict.
    data = [
        '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般',
        '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
        '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    ]
    tokenizer = JiebaTokenizer(vocab)
    data = preprocess_prediction_data(data, tokenizer)

    # Seperates data into some batches.
    batches = [
        data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_token_id),  # input_ids
        Stack(dtype="int64"),  # seq len
    ): [data for data in fn(samples)]

    results = []
    model.eval()
    for batch in batches:
        texts, seq_lens = batchify_fn(batch)
        texts = paddle.to_tensor(texts)
        seq_lens = paddle.to_tensor(seq_lens)
        logits = model(texts, seq_lens)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    print(results)
    # return results


if __name__ == "__main__":
    paddle.set_device(args.device)
    set_seed()
    label_list = ["0", "1"]
    train_path = "/home/mesie/python/data/nlp/ChnSentiCorp/train.tsv"
    dev_path = "/home/mesie/python/data/nlp/ChnSentiCorp/dev.tsv"
    test_path = "/home/mesie/python/data/nlp/ChnSentiCorp/test.tsv"
    vocab_path = '/home/mesie/python/data/nlp/senta_word_dict.txt'
    # train(vocab_path, train_path, dev_path)
    predict(vocab_path, label_list)




