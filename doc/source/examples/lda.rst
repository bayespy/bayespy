..
   Copyright (C) 2015 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


Latent Dirichlet allocation
===========================

Latent Dirichlet allocation is a widely used topic model.  The data is a
collection of documents which contain words.  The goal of the analysis is to
find topics (distribution of words in topics) and document topics (distribution
of topics in documents).


Data
----

The data consists of two vectors of equal length.  The elements in these vectors
correspond to the words in all documents combined.  If there were :math:`M`
documents and each document had :math:`K` words, the vectors contain :math:`M
\cdot K` elements.  Let :math:`M` be the number of documents in total.  The
first vector gives each word a document index :math:`i\in \{0,\ldots,M-1\}`
defining to which document the word belongs.  Let :math:`N` be the size of the
whole available vocabulary.  The second vector gives each word a vocabulary
index :math:`j\in \{0,\ldots,N-1\}` defining which word it is from the
vocabulary.

For this demo, we will just generate an artificial dataset for simplicity.  We
use the LDA model itself to generate the dataset.  First, import relevant
packages:

>>> import numpy as np
>>> from bayespy import nodes

Let us decide the number of documents and the number of words in those documents:

>>> n_documents = 10
>>> n_words = 10000

Randomly choose into which document each word belongs to:

>>> word_documents = nodes.Categorical(np.ones(n_documents)/n_documents,
...                                    plates=(n_words,)).random()

Let us also decide the size of our vocabulary:

>>> n_vocabulary = 100

Also, let us decide the true number of topics:

>>> n_topics = 5

Generate some random distributions for the topics in each document:

>>> p_topic = nodes.Dirichlet(1e-1*np.ones(n_topics),
...                           plates=(n_documents,)).random()

Generate some random distributions for the words in each topic:

>>> p_word = nodes.Dirichlet(1e-1*np.ones(n_vocabulary),
...                          plates=(n_topics,)).random()

Sample topic assignments for each word in each document:

>>> topic = nodes.Categorical(p_topic[word_documents],
...                           plates=(n_words,)).random()

And finally, draw vocabulary indices for each word in all the documents:

>>> corpus = nodes.Categorical(p_word[topic],
...                            plates=(n_words,)).random()

Now, our dataset consists of ``word_documents`` and ``corpus``, which define the
document and vocabulary indices for each word in our dataset.

.. todo::

   Use some large real-world dataset, for instance, Wikipedia.


Model
-----

Variable for learning the topic distribution for each document:

>>> p_topic = nodes.Dirichlet(np.ones(n_topics),
...                           plates=(n_documents,),
...                           name='p_topic')

Variable for learning the word distribution for each topic:

>>> p_word = nodes.Dirichlet(np.ones(n_vocabulary),
...                          plates=(n_topics,),
...                          name='p_word')

The document indices for each word in the corpus:

>>> from bayespy.inference.vmp.nodes.categorical import CategoricalMoments
>>> document_indices = nodes.Constant(CategoricalMoments(n_documents), word_documents,
...                                   name='document_indices')

Variable for learning the topic assignments of each word in the corpus:

>>> topics = nodes.Categorical(nodes.Gate(document_indices, p_topic),
...                            plates=(len(corpus),),
...                            name='topics')

The vocabulary indices for each word in the corpus:

>>> words = nodes.Categorical(nodes.Gate(topics, p_word),
...                           name='words')


Inference
---------

Observe the corpus:

>>> words.observe(corpus)

Break symmetry by random initialization:

>>> p_topic.initialize_from_random()
>>> p_word.initialize_from_random()

Construct inference engine:

>>> from bayespy.inference import VB
>>> Q = VB(words, topics, p_word, p_topic, document_indices)

Run the VB learning algorithm:

>>> Q.update(repeat=1000)
Iteration ...


Results
-------

Use ``bayespy.plot`` to plot the results:

>>> import bayespy.plot as bpplt

Plot the topic distributions for each document:

>>> bpplt.pyplot.figure()
<matplotlib.figure.Figure object at 0x...>
>>> bpplt.hinton(Q['p_topic'])
>>> bpplt.pyplot.title("Posterior topic distribution for each document")
<matplotlib.text.Text object at 0x...>
>>> bpplt.pyplot.xlabel("Topics")
<matplotlib.text.Text object at 0x...>
>>> bpplt.pyplot.ylabel("Documents")
<matplotlib.text.Text object at 0x...>

Plot the word distributions for each topic:

>>> bpplt.pyplot.figure()
<matplotlib.figure.Figure object at 0x...>
>>> bpplt.hinton(Q['p_word'])
>>> bpplt.pyplot.title("Posterior word distributions for each topic")
<matplotlib.text.Text object at 0x...>
>>> bpplt.pyplot.xlabel("Words")
<matplotlib.text.Text object at 0x...>
>>> bpplt.pyplot.ylabel("Topics")
<matplotlib.text.Text object at 0x...>

.. todo::

   Create more illustrative plots.



Stochastic variational inference
--------------------------------

LDA is a popular example for stochastic variational inference (SVI).  Using SVI
for LDA is quite simple in BayesPy.  In SVI, only a subset of the dataset is
used at each iteration step but this subset is "repeated" to get the same size
as the original dataset.  Let us define a size for the subset:

>>> subset_size = 1000

Thus, our subset will be repeat this many times:

>>> plates_multiplier = n_words / subset_size

Note that this multiplier doesn't need to be an integer.

Now, let us repeat the model construction with only one minor addition.  The
following variables are identical to previous:

>>> p_topic = nodes.Dirichlet(np.ones(n_topics),
...                           plates=(n_documents,),
...                           name='p_topic')
>>> p_word = nodes.Dirichlet(np.ones(n_vocabulary),
...                          plates=(n_topics,),
...                          name='p_word')

The document indices vector is now a bit shorter, using only a subset:

>>> document_indices = nodes.Constant(CategoricalMoments(n_documents),
...                                   word_documents[:subset_size],
...                                   name='document_indices')

Note that at this point, it doesn't matter which elements we chose for the
subset.  For the topic assignments of each word in the corpus we need to use
``plates_multiplier`` because these topic assignments for the subset are
"repeated" to recover the full dataset:

>>> topics = nodes.Categorical(nodes.Gate(document_indices, p_topic),
...                            plates=(subset_size,),
...                            plates_multiplier=(plates_multiplier,),
...                            name='topics')

Finally, the vocabulary indices for each word in the corpus are constructed as
before:

>>> words = nodes.Categorical(nodes.Gate(topics, p_word),
...                           name='words')

This node inherits the plates and multipliers from its parent ``topics``, so
there is no need to define them here.  Again, break symmetry by random
initialization:

>>> p_topic.initialize_from_random()
>>> p_word.initialize_from_random()

Construct inference engine:

>>> from bayespy.inference import VB
>>> Q = VB(words, topics, p_word, p_topic, document_indices)

In order to use SVI, we need to disable some lower bound checks, because the
lower bound doesn't anymore necessarily increase at each iteration step:

>>> Q.ignore_bound_checks = True

For the stochastic gradient ascent, we'll define some learning parameters:

>>> delay = 1
>>> forgetting_rate = 0.7

Run the inference:

>>> for n in range(1000):
...     # Observe a random mini-batch
...     subset = np.random.choice(n_words, subset_size)
...     Q['words'].observe(corpus[subset])
...     Q['document_indices'].set_value(word_documents[subset])
...     # Learn intermediate variables
...     Q.update('topics')
...     # Set step length
...     step = (n + delay) ** (-forgetting_rate)
...     # Stochastic gradient for the global variables
...     Q.gradient_step('p_topic', 'p_word', scale=step)
Iteration 1: ...

If one is interested, the lower bound values during the SVI algorithm can be plotted as:

>>> bpplt.pyplot.figure()
<matplotlib.figure.Figure object at 0x...>
>>> bpplt.pyplot.plot(Q.L)
[<matplotlib.lines.Line2D object at 0x...>]

The other results can be plotted as before.
