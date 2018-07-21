# PyTorch Seq2Seq

This repo contains tutorials covering understanding and implementing sequence-to-sequence (seq2seq) models using [PyTorch](https://github.com/pytorch/pytorch) 0.4 and [TorchText](https://github.com/pytorch/text) 0.2.3 using Python 3.6.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-sentiment-analysis/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install TorchText:

``` bash
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install both the English and German models with:

``` bash
python -m spacy download en
python -m spacy download de
```

## Tutorials

* 1 - [Sequence to Sequence Learning with Neural Networks]() **IN PROGRESS**

    Implementation of [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)

* 2 - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation] **IN PROGRESS**

    Implementation of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

* 3 - [Neural Machine Translation by Jointly Learning to Align and Translate]() **TODO**

    Implementation of [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)