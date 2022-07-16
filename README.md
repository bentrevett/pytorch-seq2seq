# PyTorch Seq2Seq

## Note: This repo only works with torchtext 0.9 or above which requires PyTorch 1.8 or above. If you are using torchtext 0.8 then please use [this](https://github.com/bentrevett/pytorch-seq2seq/tree/torchtext08) branch

This repo contains tutorials covering understanding and implementing sequence-to-sequence (seq2seq) models using [PyTorch](https://github.com/pytorch/pytorch) 1.8, [torchtext](https://github.com/pytorch/text) 0.9 and [spaCy](https://spacy.io/) 3.0,  using Python 3.8.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-seq2seq/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install torchtext:

``` bash
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install both the English and German models with:

``` bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Tutorials

* 1 - [Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)

    This first tutorial covers the workflow of a PyTorch with torchtext seq2seq project. We'll cover the basics of seq2seq networks using encoder-decoder models, how to implement these models in PyTorch, and how to use torchtext to do all of the heavy lifting with regards to text processing. The model itself will be based off an implementation of [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), which uses multi-layer LSTMs.

* 2 - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb)

    Now we have the basic workflow covered, this tutorial will focus on improving our results. Building on our knowledge of PyTorch and torchtext gained from the previous tutorial, we'll cover a second second model, which helps with the information compression problem faced by encoder-decoder models. This model will be based off an implementation of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), which uses GRUs.

* 3 - [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)

    Next, we learn about attention by implementing [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). This further allievates the information compression problem by allowing the decoder to "look back" at the input sentence by creating context vectors that are weighted sums of the encoder hidden states. The weights for this weighted sum are calculated via an attention mechanism, where the decoder learns to pay attention to the most relevant words in the input sentence.

* 4 - [Packed Padded Sequences, Masking, Inference and BLEU](https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb)

    In this notebook, we will improve the previous model architecture by adding *packed padded sequences* and *masking*. These are two methods commonly used in NLP. Packed padded sequences allow us to only process the non-padded elements of our input sentence with our RNN. Masking is used to force the model to ignore certain elements we do not want it to look at, such as attention over padded elements. Together, these give us a small performance boost. We also cover a very basic way of using the model for inference, allowing us to get translations for any sentence we want to give to the model and how we can view the attention values over the source sequence for those translations. Finally, we show how to calculate the BLEU metric from our translations.

* 5 - [Convolutional Sequence to Sequence Learning](https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb)

    We finally move away from RNN based models and implement a fully convolutional model. One of the downsides of RNNs is that they are sequential. That is, before a word is processed by the RNN, all previous words must also be processed. Convolutional models can be fully parallelized, which allow them to be trained much quicker. We will be implementing the [Convolutional Sequence to Sequence](https://arxiv.org/abs/1705.03122) model, which uses multiple convolutional layers in both the encoder and decoder, with an attention mechanism between them.  

* 6 - [Attention Is All You Need](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)

    Continuing with the non-RNN based models, we implement the Transformer model from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This model is based soley on attention mechanisms and introduces Multi-Head Attention. The encoder and decoder are made of multiple layers, with each layer consisting of Multi-Head Attention and Positionwise Feedforward sublayers. This model is currently used in many state-of-the-art sequence-to-sequence and transfer learning tasks.

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

- https://github.com/spro/practical-pytorch
- https://github.com/keon/seq2seq
- https://github.com/pengshuang/CNN-Seq2Seq
- https://github.com/pytorch/fairseq
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
