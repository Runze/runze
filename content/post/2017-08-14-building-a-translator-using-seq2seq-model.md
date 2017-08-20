---
title: First attempt at building a language translator
author: Runze
date: '2017-08-14'
slug: first-attempt-at-building-a-language-translator
categories:
  - Data analysis
tags:
  - Deep Learning
  - RNN
  - LSTM
  - NLP
description: 'Using a sequence-to-sequence model'
draft: yes
topics: []
---

### Background

After having tried my hands on LSTM and built a [text generater](https://runze.github.io/2017/07/27/when-jane-austen-oscar-wilde-and-f-scott-fitzgerald-walk-into-a-bar/), I became interested in the sequence-to-sequence models, particularly their applications in language translations. It all started with this TensorFlow [tutorial](https://www.tensorflow.org/tutorials/seq2seq) where the authors demonstrated how they built an English-to-French translator using such a model and successfully translated "Who is the president of the United States?" into French with the correct grammar ("Qui est le président des États-Unis?"). Although one can argue that, given that the model was trained on [the Europarl corpus](http://www.statmt.org/wmt15/translation-task.html) extracted from the proceedings of the European Parliament, it is not particularly difficult because such phrases as "the president of United States" should presumably be very common (and the fact that this example is a word-to-word translation with no inversion involved^[I'm curious to see how the model handles direct and indirect object pronouns.]), it is still no small feat in my opinion and its potential is obvious. In fact, an enhanced version of the model (also based on RNN) is being used to power today's [Google Translate](https://research.googleblog.com/2016/09/a-neural-network-for-machine.html).

As all TensorFlow tutorials, there is an accompanying [repo](https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate) with the full code. However, I wanted to learn other deep learning libraries, particularly [Keras](https://keras.io/). Keras is a high-level API on top of a backend system such as TensorFlow and is thereby much easier to use for prototyping purposes and has syntax closer to scikit-learn.^[And what a great [name](https://keras.io/#why-this-name-keras) it has!] In researching sequence-to-sequence models created in Keras, I came across this very helpful blog [post](https://chunml.github.io/ChunML.github.io/project/Sequence-To-Sequence/) by Trung Tran where he laid out all the components of the model so clearly that even I understood. Tran also graciously provided his full [code](https://github.com/ChunML/seq2seq), which I happily stole to experiment with.

Before moving on, it is necessary to note that the applications of sequence-to-sequence models go way beyond language translations - anything that takes a variable-length sequence and outputs another can theoretically be achieved by such models. For example, one can use it to build a chatbot, either retrieval-based or generative, as explained in this blog [post](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/) by Denny Britz, the developer of Google's encoder-decoder framework (though given the endless possibilities of the outcome in human conversations, it is much harder to implement).

The remaining of the post walks through my project of building a simple English-to-French translator. I chose translations to French because it's a language I'm actively learning and, hence, it's easy for me to assess the quality of the translations.^[As a bonus point, because the training data comes from the European Parliament, I have also picked up a few formal expressions in French :-)] My code for this project, written in Jupyter Notebooks, is saved in this [repo](https://github.com/Runze/seq2seq-translation).

### Training data

An example pair of my training data looks like this:

- English: "Resumption of the session"
- French: "Reprise de la session"

To feed such data into the model, we need to first tokenize the raw corpus into sentences and then sentences into words:

```python
from keras.preprocessing.text import text_to_word_sequence

# Split text into sentences and sentences into words
# `X_data` is the English corpus, used as the training features
# `y_data` is the French corpus, used as the training labels
X = [text_to_word_sequence(sentence) for sentence in X_data.split('\n')]
y = [text_to_word_sequence(sentence) for sentence in y_data.split('\n')]
```

On top of that, I also removed sentences that have 0 length or are longer than 50 words, which is about the 95th percentile of all data. Finally, due to the time (and the AWS cost) it would take to train a model using nearly 2 million sentences, I randomly sampled 100,000 of them for training. You can see these additional details in my [code](https://github.com/Runze/seq2seq-translation/blob/master/translate.ipynb).

### Model structure

There are different ways to create a sequence-to-sequence model and one popular solution is to use RNN with *attention mechanism*, as this encoder-decoder [framework](https://github.com/google/seq2seq) developed and open-sourced by Google. To understand why attention is superior, let's first take a look at a simpler model without it:

<img src="https://raw.githubusercontent.com/Runze/seq2seq-translation/master/illustrations/seq2seq.png" alt="alt text" width="750">

The structure is rather straightforward:

1. The encoder is a standard recurrent network that takes in a sentence written in the original language word by word. Note in this implementation, it discards all the hidden states' output except the last one, which should *theoretically* remembers all the important things about this whole sentence.^[For a detailed walkthrough of LSTM, please refer to this very helpful blog [post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.]
2. To feed this single output into the decoder, we need to first repeat it a few times to match the length of the output sequence written in the target language. In the illustration above, both sentences happen to have the same length, but it is usually not guaranteed. One easy to way to overcome the variable-length problem is to always repeat the encoder output to the maximum length of the target sentences, which in my case is capped at 50 words.
3. Lastly, in the decoder network, we feed these repeated encoder outputs into another RNN to output the final sequence. These outputs are then compared with the actual sentences written in the target language to compute losses.

If you feel a bit uncomfortable about only using the last encoder output and repeating it multiple times to feed them into the decoder network, you are not alone. Although it makes theoretical sense as LSTM is supposed to have a good memory of a long sentence, it is not always the case in practice. To deal with this problem, researchers have found that reversing the input sentence can improve the translations as, after the inversion, the beginning of the input sentence is now close to the beginning of the output sentence (as mentioned in this [paper](https://arxiv.org/pdf/1409.3215.pdf) by Google), but it still feels a bit iffy.

Attention mechanism, on the other hand, does not just rely on the last encoder output, but it also doesn't rely on all of them. As illustrated by Google in this [gif](https://camo.githubusercontent.com/a3c087bfca2eb44553f284ac13b19ccc27a333ed/68747470733a2f2f332e62702e626c6f6773706f742e636f6d2f2d3350626a5f64767430566f2f562d71652d4e6c365035492f41414141414141414251632f7a305f365774565774764152744d6b3069395f41744c6579794779563641493477434c63422f73313630302f6e6d742d6d6f64656c2d666173742e676966), the decoder selectively chooses what encoder outputs to focus on sequentially. This blog [post](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/) by Britz also provided a very helpful [illustration](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/12/Screen-Shot-2015-12-30-at-1.23.48-PM.png) of what attention does: for languages that share the similar word sequences, the model focuses on the encoder output produced by each input word sequentially most of the time but also takes into account the adjacent words when it needs to.

Despite its superiority, in this project, I still chose to implement my model using the first easier option because I was eager to build something quickly with Tran's code and I did not find an easy implementation of attention mechanism in Keras yet.

### Process data

Knowing the model structure, let's finish preparing the data. First, like all NLP models, we need to create a word-to-index mapping for both directions:

```python
def create_word_to_id_mapping(data, max_vocab_size=20000):
    counter = collections.Counter(np.hstack(data))
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # Pick the most common words up to `max_vocab_size` words
    count_pairs = count_pairs[:max_vocab_size]

    # Add 'ZERO' and 'UNK'
    # It is important to add 'ZERO' to the beginning to make sure it gets an index of 0
    # so that it would not interfere with existing words
    count_pairs.insert(0, ('ZERO', 0))
    count_pairs.append(('UNK', 0))

    # Create mapping for both directions
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    
    # Map words to indexes
    data_id = [[word_to_id[word] if word in word_to_id else word_to_id['UNK'] for word in sentence] for sentence in data]
    
    return word_to_id, id_to_word, data_id

X_word_to_id, X_id_to_word, X_id = create_word_to_id_mapping(X_small)
y_word_to_id, y_id_to_word, y_id = create_word_to_id_mapping(y_small)
```

The only thing to note from the snippet above is that, aside from the vocabulary in the training data, we added two special characters "ZERO" and "UNK": "ZERO" (represented by index 0) is used to pad both input and output sentences to a fixed length to deal with the variable-length problem (in my case, the length is set to 50 for both of them) and "UNK" is used to represent all out-of-vocabulary words (here the vocabulary size is capped at 20,000). Below is how we pad zeros to make them of equal length:

```python
X_id_padded = pad_sequences(X_id, maxlen=max_len, padding='post')
y_id_padded = pad_sequences(y_id, maxlen=max_len, padding='post')
```

Note I choose post-paddings so all the zeros are padded in the end, but I have read that it doesn't really matter whether we do pre- or post-paddings.

Lastly, as mentioned above, we reverse the encoder input sequence order because research has shown that doing so can improve the model performance:^[I didn't actually test it myself.]

```python
X_id_padded = np.array([sentence[::-1] for sentence in X_id_padded])
```

At the end of all the steps above, the encoder input for "resumption of the session" would look like (assuming no indexing is done) `["ZERO", "ZERO", "ZERO"..., "session", "the", "of", "resumption"]` and the decoder target for "reprise de la session" becomes `["reprise", "de", "la", "session", "ZERO", "ZERO", "ZERO", ...]`.

### Create model

Now the fun part begins: we are going to create the model structure as illustrated in the plot above. In Keras, this is very easy and elegant to do - we are basically adding network layers like building blocks with usually one line of code per layer. Keras has truly kept its promise of "[b]eing able to go from idea to result with the least possible delay!"

To start, we create a bare bone of the model by defining an empty `Sequential` model:

```python
model = Sequential()
```

#### Create encoder network

Then we set off to creating the encoder network. Instead of applying one-hot encodings to the training data, which will create a large sparse matrix, we are going to first feed the input into an embedding layer that condenses a vocabulary of 20,002 words (including "ZERO" and "UNK") into a dense vector of 1,024 length:

```python
# Add embedding layer
X_vocab_size = len(X_id_to_word)  # 20002
hidden_size = 1024

model.add(
    Embedding(
        input_dim=X_vocab_size,  # 20002
        output_dim=hidden_size,  # 1024
        input_length=max_len,    # 50
        mask_zero=True))
```

Note the argument `mask_zero=True` is telling subsequent layers to ignore zeros because they are not part of the actual vocabulary as they are only used to pad the input sequence to a fixed length.

Next, we add an LSTM layer to take in the encoder input word by word:

```python
# Add LSTM layer
model.add(LSTM(hidden_size))
```

There is an argument for `LSTM` that we left at its default value: `return_sequences=False`, which is telling the model to only return the last encoder output and discard the previous ones, just like we discussed above.

Lastly, we repeat this single output a number of times to match the maximum length of the output (which in this case happens to be the same as that of the input):

```python
# Repeat the last output of the LSTM layer to the size of the decoder input
model.add(RepeatVector(max_len))
```

#### Create decoder network

For decoder, instead of a single-layer LSTM, we can stack multiple ones together to improve the performance:

```python
# Stack LSTM layers
num_layers = 3

for _ in range(num_layers):
    model.add(LSTM(hidden_size, return_sequences=True))
```

Note, in this case, we set `return_sequences` to be `True` because we need to pass all the outputs from one intermediary layer to the next and, in the final layer, we use all of them to compute losses.

At this point, each outputted word is still represented by a word embedding. To compare it with the target sequence, we need a fully-connected layer to transform it back to the original shape of the vocabulary size:

```python
# Add dense layer to convert the LSTM output to the shape of target labels
y_vocab_size = len(y_id_to_word)  # 20002

model.add(TimeDistributed(Dense(y_vocab_size)))
```

Finally, we apply softmax to convert the output to a list of probabilities:

```python
model.add(Activation('softmax'))
```

*Et voilà* - that's all it takes to define the model! We can print the model summary to make sure it all looks right:

```python
model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding_1 (Embedding)      (None, 50, 1024)          20482048  
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 1024)              8392704   
# _________________________________________________________________
# repeat_vector_1 (RepeatVecto (None, 50, 1024)          0         
# _________________________________________________________________
# lstm_2 (LSTM)                (None, 50, 1024)          8392704   
# _________________________________________________________________
# lstm_3 (LSTM)                (None, 50, 1024)          8392704   
# _________________________________________________________________
# lstm_4 (LSTM)                (None, 50, 1024)          8392704   
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, 50, 20002)         20502050  
# _________________________________________________________________
# activation_1 (Activation)    (None, 50, 20002)         0         
# =================================================================
# Total params: 74,554,914
# Trainable params: 74,554,914
# Non-trainable params: 0
# _________________________________________________________________
```

Note the `None` in column `Output Shape` refers to the batch size, which is defined later in the actual training step. As a reminder, 50 is the maximum length for both the input and output sentences, 1,024 is the embedding size and hidden size, and 20,002 is the vocabulary size.

#### Compile

After defining the model, we compile it by defining the loss function, the optimizer, and, optionally, the evaluation metrics:

```python
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
```

You can see a list of available loss functions in Keras' [documentation](https://keras.io/losses/) and `categorical_crossentropy` is just a standard loss function for classifications.

Similarly, you can find all supported optimizers [here](https://keras.io/optimizers/). RMSprop ("Root Mean Square Propagation"), proposed by Geoffrey Hinton in these lecture [slides](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf), "[d]ivide[s] the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight." According to Keras, this is "usually a good choice for recurrent neural networks."

### Train model

Training a model in Keras is as simple as calling `fit` just like in scikit-learn. My code below may look a bit more involved because, due to memory constraint, I cannot apply one-hot encodings to the target matrix `y_id_padded` all at once.^[`categorical_crossentropy` requires the target to be one-hot encoded.] Hence, I had to split them into a bunch of small batches with 1,000 sentences each and train one batch at a time. This is equivalent to batching in training except the split is done manually. In addition, after each epoch, I printed out the translations of a few pre-defined sentences to get a sense of the current quality, which I found to be more intuitive than the loss figures.

```python
epochs = 60

# Due to memory limit, which affects one-hot encoding of the target sequences,
# only train 1000 sequences at a time
seq_per_iter = 1000

# Train model `epochs` times
for i in range(epochs):
    # Shuffle the training data every epoch to avoid local minima
    np.random.seed(i)
    ix = np.arange(len(X_id_padded))
    np.random.shuffle(ix)
    
    X_id_padded, y_id_padded = X_id_padded[ix], y_id_padded[ix]
    
    for j in range(len(X_id_padded) / seq_per_iter):        
        # Slice input data as explained above
        start = j * seq_per_iter
        end = min(((j + 1) * seq_per_iter), len(X_id_padded))
        
        X_id_padded_tmp = np.array(X_id_padded[start:end])
        y_id_padded_tmp = np.array(y_id_padded[start:end])
        
        # `vectorize_sentences` is a custom function used to perform one-hot encoding (omitted here)
        # I could've also used Keras built-in function `to_categorical` had I known it beforehand
        y_id_padded_tmp_vectorized = vectorize_sentences(y_id_padded_tmp, y_vocab_size)
        
        # Fit model
        # `epochs=1` because we are manually iterating the whole training dataset
        # `epochs` number of times in this for loop
        model.fit(X_id_padded_tmp, y_id_padded_tmp_vectorized, batch_size=100, epochs=1, verbose=2)
    
    # Save weights after each epoch
    model.save_weights(os.path.join(save_path, 'checkpoint_epoch_{}.hdf5'.format(i)))
    
    # Apply model to test sentences to translate
    # `sentences_to_translate_id_padded` is a few pre-defined test sentences in English
    # They have been indexed, padded, and reversed
    predictions = model.predict(np.array(sentences_to_translate_id_padded))
    
    # Greedy decoder: pick the prediction with the largest propability
    predictions = np.argmax(predictions, axis=2)
    
    # Convert predictions to words
    predictions_in_words = [' '.join([y_id_to_word[p] for p in prediction if p > 0]) for prediction in predictions]
    for k, p in enumerate(predictions_in_words):
        print 'Translation of', sentences_to_translate[k], ':', p

# Save final model
model.save(os.path.join(save_path, 'model.h5'))
```

### Result

So how did my first translation model turn out? Unfortunately, not very good. After training nearly 60 epochs over two days (and hitting my AWS budget^[Speaking of AWS, I followed this [tutorial](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html) to run Jupyter Notebooks on AWS.]), the translations I got for my test sentences are still far from usable. Below are a few examples:

```python
sentences_to_translate = [
    'i',
    'you',
    'we',
    'europe',
    'president',
    'hello',
    'resumption of the session',
    'we need a new initiative from the commission on this'
]

# Convert sentences into lists of word indexes
sentences_to_translate_words = [text_to_word_sequence(sentence) for sentence in sentences_to_translate]
sentences_to_translate_id = [[X_word_to_id[word] for word in sentence] for sentence in sentences_to_translate_words]

# Pad sentences to maximum length and reverse them
sentences_to_translate_id_padded = pad_sequences(sentences_to_translate_id, maxlen=max_len, padding='post')
sentences_to_translate_id_padded = [sentence[::-1] for sentence in sentences_to_translate_id_padded]

# Generate predictions
predictions = model.predict(np.array(sentences_to_translate_id_padded))
predictions = np.argmax(predictions, axis=2)

predictions_in_words = [' '.join([y_id_to_word[p] for p in prediction if p > 0]) for prediction in predictions]
for k, p in enumerate(predictions_in_words):
    print 'Translation of', sentences_to_translate[k], ':', p
```

```python
# Translation of i : je ne que que que
# Translation of you : vous vous
# Translation of we : nous avons nous que
# Translation of europe : l'europe nous nous
# Translation of president : monsieur président la la de
# Translation of hello : les est de de
# Translation of resumption of the session : reprise à la la
# Translation of we need a new initiative from the commission on this : nous avons besoin de
```

As you can see, in the first few single-word examples, the model was able to translate the word itself (e.g., "I" to "je") but it didn't know when to stop and still outputted random words afterwards. This is probably partially due to that I didn't have a clear stop signal for the model and padding a special end-of-sentence character in the training sequences may help.

However, it couldn't get simple expressions like "hello" right. To be fair, the Europarl data I used for training does not have many examples that use interjections. In fact, there is no use of *bonjour* in the French sentences and, as for *hello* in English, there are only two sentences that include it and they were paired with "au revoir" (goodbye) and "saluer" (to say hello), respectively.[^longnote] On the contrary, there are over 6,000 sentences with the word "president" in it. Hence, this simple model's ability is somewhat limited by its training data and does not fair well with out-of-vocabulary inputs.

[^longnote]: Specifically,

    1. The first one is "i would like to say thank you for the five years i have worked together with Vladimír Špidla because this is the last chance to say hello to him," which is paired with "je voudrais remercier Vladimír Špidla pour les cinq années durant lesquelles j'ai travaillé à ses côtés car c'est ma dernière occasion de lui dire au revoir."
    2. The second is "far be it from me to fail to say hello to Wilmya or to wish Anna good evening or Ilona or Michèle," paired with "loin de moi l'idée de ne pas saluer Wilmya de ne pas souhaiter bonsoir à Anna ou à Ilona et Michèle."

Then how does it do with in-vocabulary inputs? To test that, I provided two sentences in the end that came directly from the training samples - "resumption of the session" and "we need a new initiative from the commission on this." As shown above, for these longer sentences, it is only able to get the first few words correct (e.g., "resumption" as in "resumption of the session"), which indicates its limited ability to construct proper sentences on its own. That said, I was still proud that it aptly translated "we need" to "nous avons besoin de" - I didn't learn this expression myself until a month after my French studies began!^[In the original sentence, "we need" is paired with "il nous faut ..."]

### Potential improvements

So how can we do better? I think there are at least the following things we can try:

1. Experiment with different training data. Europarl is a very rich language data set but its focus on politics and economics can potentially make the model hard to generalize (as we see above). To make it more applicable to everyday uses, we can instead try to use movie transcripts with translations.
2. In the data preparation, in addition to padding zeros, we can try to pad begin-of-sentence and end-of-sentence symbols to the decoder input as this TensorFlow [tutorial](https://www.tensorflow.org/tutorials/seq2seq) recommends. Doing so may help prevent the model from generating nonsensical words when it should have stopped.
3. Research how to implement attention mechanism easily.

*Bref*, even though I did not get the "magical" results I was hoping for, I still found this process very educational and was completely sold on Keras. Plus, I have picked up the bonus skill of sounding importantly in French, such as saying "I will conclude by expressing my grave disappointment with..." - *je terminerai en vous faisant part de ma grande déception en ce qui concerne ...* !
