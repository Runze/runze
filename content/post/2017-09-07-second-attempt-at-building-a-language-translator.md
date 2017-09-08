---
title: Second attempt at building a language translator
author: Runze
date: '2017-09-07'
slug: second-attempt-at-building-a-language-translator
categories:
  - Data analysis
tags:
  - Deep Learning
  - RNN
  - LSTM
  - NLP
description: 'By adding attention'
draft: yes
topics: []
---

### Background

A few weeks ago, I experimented with building [a language translator](https://runze.github.io/2017/08/14/first-attempt-at-building-a-language-translator/) using a simple sequence-to-sequence model that forms a encoder-decoder structure. Since then, I have been itchy at adding an extra attention layer to it that I have been reading so much about. After many, many research, I came across (quite accidentally!) this MOOC [series](http://course.fast.ai/part2.html) offered by [fast.ai](http://www.fast.ai/), where on [Lesson 13](http://course.fast.ai/lessons/lesson13.html), instructor Jeremy Howard walked the students through a practical implementation of the attention mechanism using PyTorch. Given that PyTorch is another framework that I yet to learn and knowing that it is not as high level as Keras, I was initially hesitant in following the tutorial. However, after seeing Jeremy demonstrating the superior flexibility and customization of PyTorch, I decided to roll up my sleeves and learn PyTorch. Yes, you can't just write a couple of lines of code to build an out-of-box model in PyTorch as you can do in Keras, but when it comes to implementing a new custom layer, this is when PyTorch comes to shine.

Like my other posts, the remainder of this one will walk through the key parts of the model building process and the results. My full code, heavily borrowed from Jeremy's [tutorial](https://github.com/fastai/courses/blob/master/deeplearning2/translate-pytorch.ipynb), is hosted in this [Jupyter Notebook](https://github.com/Runze/seq2seq-translation-attention/blob/master/translate.ipynb).

### Attention layer

First of all, let's get familiar with the attention mechanism. After reading many blogposts on the subject, I gained a pretty good intuition on the idea but was still hazy about the implementation, so I decided to bite the bullet and go back to the original [paper](https://arxiv.org/abs/1409.0473) by Bahdanau et al., which was in fact surprisingly easy to understand. Yet I still need to see some actual code to see the implementation details, which is where Jeremy's tutorial helps the most. After reading through his code and writing my own, below is my understanding of how attention works (using notations from a paper by Vinyals, et. al., which I found more intuitive):



### References

Dzmitry Bahdanau,Kyunghyun Cho,and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*, 2014.

Vinyals, O., L. Kaiser, T. Koo, S. Petrov, I. Sutskever & G. E. Hinton (2014). Grammar
as a foreign language. *CoRR, abs/1412.7449.*
