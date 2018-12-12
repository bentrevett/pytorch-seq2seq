# PyTorch Seq2Seq

This repo contains tutorials covering understanding and implementing sequence-to-sequence (seq2seq) models using [PyTorch](https://github.com/pytorch/pytorch) 1.0 and [TorchText](https://github.com/pytorch/text) 0.3 using Python 3.6.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-seq2seq/issues/new). I welcome any feedback, positive or negative!**

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

* 1 - [Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)

    This first tutorial covers the workflow of a PyTorch with TorchText seq2seq project. We'll cover the basics of seq2seq networks using encoder-decoder models, how to implement these models in PyTorch, and how to use TorchText to do all of the heavy lifting with regards to text processing. The model itself will be based off an implementation of [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), which uses multi-layer LSTMs.

* 2 - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb)

    Now we have the basic workflow covered, this tutorial will focus on improving our results. Building on our knowledge of PyTorch and TorchText gained from the previous tutorial, we'll cover a second second model, which helps with the "information compression" problem faced by encoder-decoder models. This model will be based off an implementation of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), which uses GRUs.

* 3 - [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)

    Finally, we learn about attention by implementing [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). This further allievates the "information compression" problem by allowing the decoder to "look back" at the input sentence by creating context vectors that are weighted sums of the encoder hidden states. The weights for this weighted sum are calculated via an attention mechanism, where the decoder learns to pay attention to the most relevant words in the input sentence.
