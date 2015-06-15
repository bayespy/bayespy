################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np

from bayespy import nodes
from bayespy.inference import VB
from bayespy.inference.vmp.nodes.constant import Constant
from bayespy.inference.vmp.nodes.categorical import CategoricalMoments
import bayespy.plot as bpplt


def model(n_documents, n_topics, n_vocabulary, corpus, word_documents, plates_multiplier=1):
    '''
    Construct Latent Dirichlet Allocation model.
    
    Parameters
    ----------
    
    documents : int
        The number of documents

    topics : int
        The number of topics

    vocabulary : int
        The number of words in the vocabulary

    corpus : integer array
        The vocabulary index of each word in the corpus

    word_documents : integer array
        The document index of each word in the corpus
    '''

    # Topic distributions for each document
    p_topic = nodes.Dirichlet(np.ones(n_topics),
                              plates=(n_documents,),
                              name='p_topic')

    # Word distributions for each topic
    p_word = nodes.Dirichlet(np.ones(n_vocabulary),
                             plates=(n_topics,),
                             name='p_word')

    # Use a simple wrapper node so that the value of this can be changed if one
    # uses stocahstic variational inference
    word_documents = Constant(CategoricalMoments(n_documents), word_documents,
                              name='word_documents')

    # Choose a topic for each word in the corpus
    topics = nodes.Categorical(nodes.Gate(word_documents, p_topic),
                               plates=(len(corpus),),
                               plates_multiplier=(plates_multiplier,),
                               name='topics')

    # Choose each word in the corpus from the vocabulary
    words = nodes.Categorical(nodes.Gate(topics, p_word),
                              name='words')

    # Observe the corpus
    words.observe(corpus)

    # Break symmetry by random initialization
    p_topic.initialize_from_random()
    p_word.initialize_from_random()

    return VB(words, topics, p_word, p_topic, word_documents)


def generate_data(n_documents, n_topics, n_vocabulary, n_words):

    # Generate random data from the generative model

    # Generate document assignments for the words
    word_documents = nodes.Categorical(np.ones(n_documents)/n_documents,
                                       plates=(n_words,)).random()

    # Topic distribution for each document
    p_topic = nodes.Dirichlet(1e-1*np.ones(n_topics),
                              plates=(n_documents,)).random()

    # Word distribution for each topic
    p_word = nodes.Dirichlet(1e-1*np.ones(n_vocabulary),
                             plates=(n_topics,)).random()

    # Topic for each word in each document
    topic = nodes.Categorical(p_topic[word_documents],
                              plates=(n_words,)).random()

    # Each word in each document
    corpus = nodes.Categorical(p_word[topic],
                               plates=(n_words,)).random()

    bpplt.pyplot.figure()
    bpplt.hinton(p_topic)
    bpplt.pyplot.title("True topic distribution for each document")
    bpplt.pyplot.xlabel("Topics")
    bpplt.pyplot.ylabel("Documents")

    bpplt.pyplot.figure()
    bpplt.hinton(p_word)
    bpplt.pyplot.title("True word distributions for each topic")
    bpplt.pyplot.xlabel("Words")
    bpplt.pyplot.ylabel("Topics")

    return (corpus, word_documents)


def run(n_documents=30, n_topics=5, n_vocabulary=10, n_words=50000, stochastic=False, maxiter=1000, seed=None):

    if seed is not None:
        np.random.seed(seed)

    (corpus, word_documents) = generate_data(n_documents, n_topics, n_vocabulary, n_words)

    if not stochastic:

        Q = model(n_documents=n_documents, n_topics=n_topics, n_vocabulary=n_vocabulary,
                  corpus=corpus, word_documents=word_documents)

        Q.update(repeat=maxiter)

    else:

        subset_size = 1000

        Q = model(n_documents=n_documents, n_topics=n_topics, n_vocabulary=n_vocabulary,
                  corpus=corpus[:subset_size], word_documents=word_documents[:subset_size],
                  plates_multiplier=n_words/subset_size)

        Q.ignore_bound_checks = True
        delay = 1
        forgetting_rate = 0.7
        for n in range(maxiter):

            # Observe a mini-batch
            subset = np.random.choice(n_words, subset_size)
            Q['words'].observe(corpus[subset])
            Q['word_documents'].set_value(word_documents[subset])

            # Learn intermediate variables
            Q.update('topics')

            # Set step length
            step = (n + delay) ** (-forgetting_rate)

            # Stochastic gradient for the global variables
            Q.gradient_step('p_topic', 'p_word', scale=step)

        bpplt.pyplot.figure()
        bpplt.pyplot.plot(Q.L)


    bpplt.pyplot.figure()
    bpplt.hinton(Q['p_topic'])
    bpplt.pyplot.title("Posterior topic distribution for each document")
    bpplt.pyplot.xlabel("Topics")
    bpplt.pyplot.ylabel("Documents")

    bpplt.pyplot.figure()
    bpplt.hinton(Q['p_word'])
    bpplt.pyplot.title("Posterior word distributions for each topic")
    bpplt.pyplot.xlabel("Words")
    bpplt.pyplot.ylabel("Topics")

    return


if __name__ == '__main__':
    import sys, getopt, os
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "",
                                   ["documents=",
                                    "topics=",
                                    "vocabulary=",
                                    "words=",
                                    "stochastic",
                                    "seed=",
                                    "maxiter="])
    except getopt.GetoptError:
        print('python lda.py <options>')
        print('--documents=<INT>   The number of documents')
        print('--topics=<INT>      The number of topics')
        print('--vocabulary=<INT>  The size of the vocabulary')
        print('--words=<INT>       The size of the corpus')
        print('--maxiter=<INT>     Maximum number of VB iterations')
        print('--seed=<INT>        Seed (integer) for the RNG')
        print('--stochastic        Use stochastic variational inference')
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt == "--documents":
            kwargs["n_documents"] = int(arg)
        elif opt == "--topics":
            kwargs["n_topics"] = int(arg)
        elif opt == "--vocabulary":
            kwargs["n_vocabulary"] = int(arg)
        elif opt == "--words":
            kwargs["n_words"] = int(arg)
        elif opt == "--stochastic":
            kwargs["stochastic"] = True

    #raise NotImplementedError("Work in progress.. This demo is not yet finished")
    run(**kwargs)
    bpplt.pyplot.show()

