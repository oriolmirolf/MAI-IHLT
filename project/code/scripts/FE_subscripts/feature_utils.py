import nltk
from nltk.corpus import brown, wordnet as wn, wordnet_ic

import numpy as np

word_freq = nltk.FreqDist(w.lower() for w in brown.words())
total_freq = sum(word_freq.values())
brown_ic = wordnet_ic.ic('ic-brown.dat')


def jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets.

    Args:
        set1 (set): First set.
        set2 (set): Second set.

    Returns:
        float: Jaccard similarity.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def containment_similarity(set1, set2):
    """
    Compute containment similarity between two sets.

    Args:
        set1 (set): First set.
        set2 (set): Second set.

    Returns:
        float: Containment similarity.
    """
    return len(set1.intersection(set2)) / len(set1) if set1 else 0.0

def information_content(w):
    """
    Compute the information content of a word.

    Args:
        w (str): Word.

    Returns:
        float: Information content.
    """
    freq = word_freq[w.lower()] if w.lower() in word_freq else 1
    ic = np.log(total_freq / freq)
    return ic

def harmonic_mean(a, b):
    """
    Compute the harmonic mean of two numbers.

    Args:
        a (float): First number.
        b (float): Second number.

    Returns:
        float: Harmonic mean.
    """
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0

def wordnet_similarity(w1, w2):
    """
    Compute the maximum WordNet similarity between two words using Resnik's measure.

    Args:
        w1 (str): First word.
        w2 (str): Second word.

    Returns:
        float: Maximum similarity between two words.
    """
    synsets1 = wn.synsets(w1)
    synsets2 = wn.synsets(w2)
    max_sim = 0.0
    for s1 in synsets1:
        for s2 in synsets2:
            if s1.pos() == s2.pos() and s1.pos() in ('n', 'v'):
                sim = s1.res_similarity(s2, brown_ic)
                if sim and sim > max_sim:
                    max_sim = sim
    return max_sim