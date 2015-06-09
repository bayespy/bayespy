################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np

from bayespy import nodes
from bayespy.inference import VB
import bayespy.plot as bpplt


def model(documents, topics, vocabulary, corpus, word_documents):
    '''
    Construct Latent Dirichlet Allocation model.
    
    Parameters
    ----------
    
    documents : int
        The number of documents

    topics : int
        The number of topics

    vocabulary : int
        The number of corpus in the vocabulary

    corpus : integer array
        The vocabulary index of each word in the corpus

    word_documents : integer array
        The document index of each word in the corpus
    '''

    # Topic distributions for each document
    p_topic = nodes.Dirichlet(np.ones(topics),
                              plates=(documents,))

    # Word distributions for each topic
    p_word = nodes.Dirichlet(np.ones(vocabulary),
                             plates=(topics,))

    # Choose a topic for each word in the corpus
    topic = nodes.Categorical(nodes.Gate(word_documents, p_topic),
                              plates=(len(corpus),))

    # Choose each word in the corpus from the vocabulary
    word = nodes.Categorical(nodes.Gate(topic, p_word),
                             plates=(len(corpus),))

    # Observe the corpus
    word.observe(corpus)

    return VB(word, topic, p_word, p_topic)


def generate_data(documents, topics, vocabulary, words):

    # Generate random data from the generative model
    
    word_documents = nodes.Categorical(np.ones(documents)/documents).random()

    p_topic = nodes.Dirichlet(np.ones(topics),
                              plates=(documents,)).random()
    p_word = nodes.Dirichlet(np.ones(vocabulary),
                             plates=(topics,)).random()
    topic = nodes.Categorical(p_topic[word_documents],
                              plates=(words,)).random()
    corpus = nodes.Categorical(p_word[word_documents],
                               plates(words,)).random()

    print(word_documents)
    print(corpus)

    return (corpus, word_documents)


def run(documents, topics, vocabulary, words):

    (corpus, word_documents) = generate_data(documents, topics, vocabulary, words)

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
        sys.exit(2)

    kwargs = {}
    for opt, arg in opts:
        if opt == "--maxiter":
            kwargs["maxiter"] = int(arg)
        elif opt == "--seed":
            kwargs["seed"] = int(arg)
        elif opt in ("--documents",):
            kwargs["documents"] = int(arg)
        elif opt in ("--topics",):
            kwargs["topics"] = int(arg)
        elif opt in ("--vocabulary",):
            kwargs["vocabulary"] = int(arg)
        elif opt in ("--words",):
            kwargs["words"] = int(arg)

    raise NotImplementedError("Work in progress.. This demo is not yet finished")
    run(**kwargs)
    plt.show()

