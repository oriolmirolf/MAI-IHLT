# feature_extraction/feature_utils.py

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet_ic

import spacy
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from itertools import chain
import functools

nlp = spacy.load('en_core_web_sm')

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pandas as pd

# Download required NLTK data files
nltk.download('wordnet')
nltk.download('wordnet_ic')
nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load the information content file for Resnik similarity
brown_ic = wordnet_ic.ic('ic-brown.dat')


@functools.lru_cache(maxsize=None)
def preprocess_sentence(s):
    """Preprocess a sentence by tokenizing and lowercasing."""
    tokens = nltk.word_tokenize(s.lower())
    return tuple(tokens)


@functools.lru_cache(maxsize=None)
def get_pos_tags(tokens):
    """Get POS tags for a list of tokens."""
    pos_tags = nltk.pos_tag(tokens)
    return tuple(pos_tags)


@functools.lru_cache(maxsize=None)
def get_dependency_relations(s):
    """Get dependency relations from a sentence using spaCy."""
    doc = nlp(s)
    deps = []
    for token in doc:
        if token.dep_ != 'ROOT':
            deps.append((token.dep_, token.head.text, token.text))
    return tuple(deps)


def extract_features(s1, s2):
    """Extracts lexical, syntactic, and semantic features."""
    features = {}
    from feature_extraction.lexical_features import lexical_features
    from feature_extraction.syntactic_features import syntactic_features
    from feature_extraction.semantic_features import semantic_features

    features.update(lexical_features(s1, s2))
    features.update(syntactic_features(s1, s2))
    features.update(semantic_features(s1, s2))
    return features


def greedy_string_tiling(s1, s2, min_match_length=3):
    """Compute the Greedy String Tiling similarity between two strings."""
    # Tokenize the strings
    tokens1 = s1.split()
    tokens2 = s2.split()
    matches = []
    marked1 = [False] * len(tokens1)
    marked2 = [False] * len(tokens2)
    max_match_length = min(len(tokens1), len(tokens2))

    total_matched = 0

    for match_length in range(max_match_length, min_match_length - 1, -1):
        i = 0
        while i <= len(tokens1) - match_length:
            if not any(marked1[i:i + match_length]):
                substring1 = tokens1[i:i + match_length]
                j = 0
                while j <= len(tokens2) - match_length:
                    if not any(marked2[j:j + match_length]):
                        substring2 = tokens2[j:j + match_length]
                        if substring1 == substring2:
                            # Mark the tokens
                            for k in range(match_length):
                                marked1[i + k] = True
                                marked2[j + k] = True
                            total_matched += match_length
                            i += match_length - 1  # Skip over the matched tokens
                            break
                    j += 1
            i += 1
    average_length = (len(tokens1) + len(tokens2)) / 2
    gst_similarity = total_matched / average_length if average_length != 0 else 0
    return gst_similarity


def longest_common_substring_length(s1, s2):
    """
    Compute the length of the longest common substring between two strings.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        int: Length of the longest common substring.
    """
    m = len(s1)
    n = len(s2)
    result = 0
    LCSuff = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                LCSuff[i + 1][j + 1] = LCSuff[i][j] + 1
                result = max(result, LCSuff[i + 1][j + 1])
            else:
                LCSuff[i + 1][j + 1] = 0
    return result


def ngram_overlap_ratio(tokens1, tokens2, n=2):
    """
    Calculate n-gram overlap ratio between two token lists.

    Args:
        tokens1 (list): First token list.
        tokens2 (list): Second token list.
        n (int): n-gram size.

    Returns:
        float: n-gram overlap ratio.
    """
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))
    overlap = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(overlap) / len(union) if len(union) != 0 else 0


def longest_common_subsequence(X, Y):
    """
    Compute the length of the longest common subsequence between two lists.

    Args:
        X (list): First list.
        Y (list): Second list.

    Returns:
        int: Length of the longest common subsequence.
    """
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                L[i + 1][j + 1] = L[i][j] + 1
            else:
                L[i + 1][j + 1] = max(L[i + 1][j], L[i][j + 1])
    return L[m][n]


def word_order_similarity(tokens1, tokens2):
    """
    Calculate word order similarity between two token lists using Spearman's rank correlation.

    Args:
        tokens1 (list): First token list.
        tokens2 (list): Second token list.

    Returns:
        float: Word order similarity score.
    """
    # Get the union of tokens
    all_tokens = list(set(tokens1 + tokens2))
    # Create index mapping based on the combined tokens
    token_idx = {token: idx for idx, token in enumerate(all_tokens)}

    # Create order vectors for both sentences based on the combined token index
    order_vec1 = [token_idx[token] for token in tokens1 if token in token_idx]
    order_vec2 = [token_idx[token] for token in tokens2 if token in token_idx]

    # Pad the shorter vector to match lengths
    max_len = max(len(order_vec1), len(order_vec2))
    if len(order_vec1) < max_len:
        order_vec1 += [-1] * (max_len - len(order_vec1))
    if len(order_vec2) < max_len:
        order_vec2 += [-1] * (max_len - len(order_vec2))

    # Use Spearman correlation
    if len(order_vec1) >= 2 and len(order_vec2) >= 2:
        correlation, _ = spearmanr(order_vec1, order_vec2)
        return correlation if not np.isnan(correlation) else 0
    else:
        return 0


def tree_edit_distance(s1, s2):
    """
    Compute a simplified tree edit distance between the dependency parse trees of two sentences.

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        float: Normalized tree edit distance.
    """
    # Reference: Used by [Štajner et al., 2012]
    def get_dependency_tree(sentence):
        doc = nlp(sentence)
        edges = []
        for token in doc:
            for child in token.children:
                edges.append((token.orth_, child.orth_))
        return edges

    tree1 = get_dependency_tree(s1)
    tree2 = get_dependency_tree(s2)

    # Using set difference as a simplification
    diff = len(set(tree1) ^ set(tree2))
    total = len(set(tree1) | set(tree2))
    return diff / total if total != 0 else 0


def word_similarity(w1, w2):
    """
    Compute semantic similarity between two words using WordNet.

    Returns:
        float: Maximum Wu-Palmer similarity between synsets of w1 and w2.
    """
    synsets1 = wn.synsets(w1)
    synsets2 = wn.synsets(w2)
    max_sim = 0
    for s1 in synsets1:
        for s2 in synsets2:
            sim = s1.wup_similarity(s2)
            if sim and sim > max_sim:
                max_sim = sim
    return max_sim if max_sim is not None else 0


def soft_jaccard_similarity(tokens1_set, tokens2_set):
    """
    Compute the Soft Jaccard similarity between two sets of tokens.

    Reference: [Jimenez et al., 2012] - UNAL-NLP: Combining Soft Cardinality Features
    """
    tokens1 = list(tokens1_set)
    tokens2 = list(tokens2_set)
    sim_matrix = np.zeros((len(tokens1), len(tokens2)))

    for i, token1 in enumerate(tokens1):
        for j, token2 in enumerate(tokens2):
            sim_matrix[i, j] = word_similarity(token1, token2)

    intersection = sim_matrix.max(axis=1).sum()
    union = len(tokens1) + len(tokens2) - intersection
    return intersection / union if union != 0 else 0


def avg_max_wordnet_similarity(tokens1, tokens2, similarity_func):
    """
    Compute the average and maximum word similarity between two lists of tokens using WordNet and given similarity function.

    Args:
        tokens1 (list): First list of tokens.
        tokens2 (list): Second list of tokens.
        similarity_func: Function to compute similarity between two synsets.

    Returns:
        avg_sim: Average maximum similarity for tokens in tokens1.
        max_sim: Maximum similarity across all token pairs.
    """
    total_sim = 0
    count = 0
    max_sim = 0

    for word1 in tokens1:
        synsets1 = wn.synsets(word1)
        max_sim_word = 0
        for word2 in tokens2:
            synsets2 = wn.synsets(word2)
            for syn1 in synsets1:
                for syn2 in synsets2:
                    # Only compute similarity for synsets with the same POS
                    if syn1.pos() == syn2.pos():
                        sim = similarity_func(syn1, syn2)
                        if sim is not None:
                            if sim > max_sim:
                                max_sim = sim
                            if sim > max_sim_word:
                                max_sim_word = sim
        if max_sim_word > 0:
            total_sim += max_sim_word
            count += 1
    avg_sim = total_sim / count if count > 0 else 0
    return avg_sim, max_sim

def synonym_overlap_ratio(tokens1, tokens2):
    """
    Calculate synonym overlap ratio using WordNet synonyms.
    """
    synonyms1 = set()
    synonyms2 = set()

    for token in tokens1:
        synsets = wn.synsets(token)
        synonyms1.update(set(chain.from_iterable([syn.lemma_names() for syn in synsets])))

    for token in tokens2:
        synsets = wn.synsets(token)
        synonyms2.update(set(chain.from_iterable([syn.lemma_names() for syn in synsets])))

    overlap = synonyms1.intersection(synonyms2)
    union = synonyms1.union(synonyms2)
    return len(overlap) / len(union) if len(union) else 0

def lexical_chain_overlap_ratio(tokens1, tokens2):
    """
    Calculate the overlap ratio of lexical chains between two token lists.
    Lexical chains are constructed using hypernyms and hyponyms.
    """
    chain1 = set()
    chain2 = set()
    for token in tokens1:
        synsets = wn.synsets(token)
        for syn in synsets:
            chain1.update(set(syn.lemma_names()))
            hypernyms = syn.hypernyms()
            hyponyms = syn.hyponyms()
            for h in hypernyms + hyponyms:
                chain1.update(set(h.lemma_names()))

    for token in tokens2:
        synsets = wn.synsets(token)
        for syn in synsets:
            chain2.update(set(syn.lemma_names()))
            hypernyms = syn.hypernyms()
            hyponyms = syn.hyponyms()
            for h in hypernyms + hyponyms:
                chain2.update(set(h.lemma_names()))

    overlap = chain1.intersection(chain2)
    union = chain1.union(chain2)
    return len(overlap) / len(union) if len(union) else 0

def antonym_overlap(tokens1, tokens2):
    """
    Compute the ratio of antonym pairs between two lists of tokens.

    Reference: Inspired by [Bär et al., 2012]

    Returns:
        float: Antonym overlap ratio.
    """
    antonym_pairs = 0
    total_pairs = 0
    for word1 in tokens1:
        synsets1 = wn.synsets(word1)
        antonyms1 = set()
        for syn in synsets1:
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    antonyms1.add(ant.name())
        for word2 in tokens2:
            if word2 in antonyms1:
                antonym_pairs += 1
            total_pairs += 1
    return antonym_pairs / total_pairs if total_pairs > 0 else 0

def named_entity_type_overlap(s1, s2):
    """
    Calculate named entity type overlap between two sentences.
    """
    doc1 = nlp(s1)
    doc2 = nlp(s2)

    ne_types1 = set([(ent.text.lower(), ent.label_) for ent in doc1.ents])
    ne_types2 = set([(ent.text.lower(), ent.label_) for ent in doc2.ents])

    overlap = ne_types1.intersection(ne_types2)
    union = ne_types1.union(ne_types2)
    return len(overlap) / len(union) if len(union) else 0

def simplified_lesk_similarity(tokens1, tokens2):
    """
    Simplified Lesk algorithm: count overlapping words in definitions (glosses) of the words.
    """
    glosses1 = []
    for token in tokens1:
        synsets = wn.synsets(token)
        for syn in synsets:
            glosses1.extend(syn.definition().split())

    glosses2 = []
    for token in tokens2:
        synsets = wn.synsets(token)
        for syn in synsets:
            glosses2.extend(syn.definition().split())

    overlap = set(glosses1).intersection(set(glosses2))
    union = set(glosses1).union(set(glosses2))
    return len(overlap) / len(union) if len(union) else 0

def hypernym_hyponym_overlap_ratio(tokens1, tokens2):
    """
    Calculate the ratio of words that have hypernyms or hyponyms in common.
    """
    hypernyms1 = set()
    hyponyms1 = set()
    for token in tokens1:
        synsets = wn.synsets(token)
        for syn in synsets:
            hypernyms1.update(set(chain.from_iterable([h.lemma_names() for h in syn.hypernyms()])))
            hyponyms1.update(set(chain.from_iterable([h.lemma_names() for h in syn.hyponyms()])))

    hypernyms2 = set()
    hyponyms2 = set()
    for token in tokens2:
        synsets = wn.synsets(token)
        for syn in synsets:
            hypernyms2.update(set(chain.from_iterable([h.lemma_names() for h in syn.hypernyms()])))
            hyponyms2.update(set(chain.from_iterable([h.lemma_names() for h in syn.hyponyms()])))

    hypernym_overlap = hypernyms1.intersection(hypernyms2)
    hyponym_overlap = hyponyms1.intersection(hyponyms2)
    total_overlap = hypernym_overlap.union(hyponym_overlap)
    total_union = hypernyms1.union(hypernyms2).union(hyponyms1).union(hyponyms2)
    return len(total_overlap) / len(total_union) if len(total_union) else 0

def lsa_similarity(s1, s2):
    """
    Calculate semantic similarity using Latent Semantic Analysis (LSA).
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([s1, s2])
    n_features = tfidf_matrix.shape[1]

    if n_features >= 2:  # Ensure there are enough features for meaningful analysis
        n_components = min(100, n_features - 1)
        try:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            lsa = make_pipeline(svd, Normalizer(copy=False))
            lsa_matrix = lsa.fit_transform(tfidf_matrix)
            sim = cosine_similarity([lsa_matrix[0]], [lsa_matrix[1]])[0][0]
            if np.isnan(sim):  # Handle NaN results
                sim = 0.0
        except ValueError as e:
            # Catch errors like empty matrices or insufficient rank
            sim = 0.0
    else:
        # Fallback to cosine similarity of TF-IDF vectors
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        if np.isnan(sim):  # Handle NaN results
            sim = 0.0
    return sim

def lda_similarity(s1, s2):
    """
    Calculate topic similarity using Latent Dirichlet Allocation (LDA).
    """
    count_vectorizer = CountVectorizer()
    tf_matrix = count_vectorizer.fit_transform([s1, s2])
    n_features = tf_matrix.shape[1]

    if n_features >= 2:
        n_components = min(10, n_features - 1)
        lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
        lda_matrix = lda.fit_transform(tf_matrix)
        sim = cosine_similarity([lda_matrix[0]], [lda_matrix[1]])[0][0]
        if np.isnan(sim):
            sim = 0.0
    else:
        # Not enough features to perform LDA
        sim = 0.0
    return sim

def compute_sentiment_score(tokens):
    """
    Compute average sentiment scores using SentiWordNet.

    Args:
        tokens (list): List of tokens.

    Returns:
        pos_score (float): Positive sentiment score.
        neg_score (float): Negative sentiment score.
        obj_score (float): Objective sentiment score.
    """
    pos_score = 0
    neg_score = 0
    obj_score = 0
    count = 0
    for word in tokens:
        synsets = wn.synsets(word)
        if synsets:
            syn = synsets[0]  # Take the first synset
            swn_synset = swn.senti_synset(syn.name())
            pos_score += swn_synset.pos_score()
            neg_score += swn_synset.neg_score()
            obj_score += swn_synset.obj_score()
            count +=1
    if count > 0:
        pos_score /= count
        neg_score /= count
        obj_score /= count
    return pos_score, neg_score, obj_score

def count_negations(tokens):
    """
    Count the number of negation words in a list of tokens.

    Args:
        tokens (list): List of tokens.

    Returns:
        int: Number of negation words.
    """
    negations = set(["not", "no", "never", "n't", "nothing", "nowhere", "neither", "without", "hardly", "barely", "scarcely"])
    return sum(1 for token in tokens if token.lower() in negations)

def semantic_role_overlap(s1, s2):
    """
    Compute the overlap of semantic roles between two sentences.

    Reference: [Heilman and Madnani, 2012] - ETS: Discriminative Edit Models
    """
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    srl1 = extract_semantic_roles(doc1)
    srl2 = extract_semantic_roles(doc2)
    overlap = srl1.intersection(srl2)
    union = srl1.union(srl2)
    return len(overlap) / len(union) if len(union) else 0

def extract_semantic_roles(doc):
    """
    Extract simplified semantic roles from a spaCy Doc.
    """
    roles = set()
    for token in doc:
        if token.dep_ in {'nsubj', 'dobj', 'iobj', 'pobj', 'attr'}:
            roles.add((token.lemma_, token.dep_))
    return roles

def temporal_expression_overlap(s1, s2):
    """
    Compute overlap of temporal expressions between two sentences.

    Reference: [Sultan et al., 2012] - DLS@CU: Sentence Similarity from Word Alignment
    """
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    temp_expressions1 = set([ent.text for ent in doc1.ents if ent.label_ in {'DATE', 'TIME'}])
    temp_expressions2 = set([ent.text for ent in doc2.ents if ent.label_ in {'DATE', 'TIME'}])
    overlap = temp_expressions1.intersection(temp_expressions2)
    union = temp_expressions1.union(temp_expressions2)
    return len(overlap) / len(union) if len(union) else 0