# PyTorch Seq2Seq

This repo contains tutorials covering understanding and implementing sequence-to-sequence (seq2seq) models using [PyTorch](https://github.com/pytorch/pytorch), with Python 3.9. Specifically, we'll train models to translate from German to English.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-seq2seq/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

Install the required dependencies with: `pip install -r requirements.txt --upgrade`.

We'll also make use of [spaCy](https://spacy.io/) to tokenize our data which requires installing both the English and German models with:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Tutorials

-   1 - [Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq/blob/main/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/main/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)

    This first tutorial covers the workflow of a seq2seq project with PyTorch. We'll cover the basics of seq2seq networks using encoder-decoder models, how to implement these models in PyTorch, and how to use the datasets/spacy/torchtext/evaluate libraries to do all of the heavy lifting. The model itself will be based off an implementation of [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), which uses multi-layer LSTMs.

-   2 - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://github.com/bentrevett/pytorch-seq2seq/blob/main/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/main/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb)

    Now we have the basic workflow covered, this tutorial will focus on improving our results. Building on our knowledge of PyTorch, we'll implement a second model, which helps with the information compression problem faced by encoder-decoder models. This model will be based off an implementation of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), which uses GRUs.

-   3 - [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/bentrevett/pytorch-seq2seq/blob/main/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/main/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)

    Next, we learn about attention by implementing [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). This further allievates the information compression problem by allowing the decoder to "look back" at the input sentence by creating context vectors that are weighted sums of the encoder hidden states. The weights for this weighted sum are calculated via an attention mechanism, where the decoder learns to pay attention to the most relevant words in the input sentence.

## Legacy Tutorials

Previous versions of these tutorials used features from the torchtext library which are no longer available. These are stored in the [legacy](https://github.com/bentrevett/pytorch-seq2seq/tree/main/legacy) directory.

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

-   https://github.com/spro/practical-pytorch
-   https://github.com/keon/seq2seq
-   https://github.com/pengshuang/CNN-Seq2Seq
-   https://github.com/pytorch/fairseq
-   https://github.com/jadore801120/attention-is-all-you-need-pytorch
-   http://nlp.seas.harvard.edu/2018/04/03/attention.html
-   https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
