# feature_extraction.py

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
from nltk.metrics.distance import jaro_winkler_similarity
from nltk.corpus import sentiwordnet as swn

import spacy
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from itertools import chain
import functools

nlp = spacy.load('en_core_web_sm')

# Since we need to cache computations per sentence, we define functions that process individual sentences and decorate them with lru_cache

@functools.lru_cache(maxsize=None)
def preprocess_sentence(s):
    """
    Preprocess a sentence by tokenizing and lowercasing.

    Returns:
        tuple: A tuple of tokens.
    """
    tokens = nltk.word_tokenize(s.lower())
    return tuple(tokens)

@functools.lru_cache(maxsize=None)
def get_pos_tags(tokens):
    """
    Get POS tags for a list of tokens.

    Returns:
        tuple: A tuple of (word, POS tag) pairs.
    """
    pos_tags = nltk.pos_tag(tokens)
    return tuple(pos_tags)

@functools.lru_cache(maxsize=None)
def get_dependency_relations(s):
    """
    Get dependency relations from a sentence using spaCy.

    Returns:
        tuple: A tuple of dependency relations.
    """
    doc = nlp(s)
    deps = []
    for token in doc:
        if token.dep_ != 'ROOT':
            deps.append((token.dep_, token.head.text, token.text))
    return tuple(deps)

@functools.lru_cache(maxsize=None)
def extract_features(s1, s2):
    """
    Extracts lexical, syntactic, and semantic features.

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of combined features.
    """
    features = {}
    features.update(lexical_features(s1, s2))
    features.update(syntactic_features(s1, s2))
    features.update(semantic_features(s1, s2))
    return features

def lexical_features(s1, s2):
    """
    Compute lexical similarity features between two sentences.

    Derived from methods used in SemEval 2012 papers [Baer et al., 2012], [Štajner et al., 2012], [Glinos, 2012].

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of lexical features.
    """
    tokens1 = preprocess_sentence(s1)
    tokens2 = preprocess_sentence(s2)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens1_no_stop = [w for w in tokens1 if w not in stopwords]
    tokens2_no_stop = [w for w in tokens2 if w not in stopwords]

    # Word overlap
    # Reference: Basic text similarity measures used by multiple teams in SemEval 2012
    overlap = set(tokens1_no_stop).intersection(set(tokens2_no_stop))
    union = set(tokens1_no_stop).union(set(tokens2_no_stop))
    word_overlap_ratio = len(overlap) / len(union) if len(union) != 0 else 0

    # Jaccard similarity
    # Reference: Used by multiple teams in SemEval 2012 [Baer et al., 2012], [Glinos, 2012]
    jaccard = word_overlap_ratio

    # Dice coefficient
    # Reference: Used by teams in SemEval 2012 [Baer et al., 2012]
    dice_coeff = (2 * len(overlap)) / (len(tokens1_no_stop) + len(tokens2_no_stop)) if (len(tokens1_no_stop) + len(tokens2_no_stop)) != 0 else 0

    # Overlap coefficient
    # Reference: Basic text similarity measures
    min_len = min(len(tokens1_no_stop), len(tokens2_no_stop))
    overlap_coeff = len(overlap) / min_len if min_len != 0 else 0

    # Levenshtein Distance (Edit Distance)
    # Reference: Used by [Glinos, 2012]
    edit_distance = nltk.edit_distance(' '.join(tokens1), ' '.join(tokens2))
    max_len = max(len(' '.join(tokens1)), len(' '.join(tokens2)))
    norm_edit_distance = 1 - (edit_distance / max_len) if max_len != 0 else 0

    # Jaro-Winkler similarity
    # Reference: Used by [Jimenez et al., 2012]
    jaro_winkler = jaro_winkler_similarity(' '.join(tokens1), ' '.join(tokens2))

    # Cosine similarity using TF-IDF vectors
    # Reference: Used by the UKP team [Baer et al., 2012]
    tfidf_vectorizer = TfidfVectorizer().fit([s1, s2])
    tfidf_vectors = tfidf_vectorizer.transform([s1, s2])
    cosine_sim = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]

    # Euclidean distance between TF-IDF vectors
    # Reference: Basic vector space models
    euclidean_dist = np.linalg.norm(tfidf_vectors[0].toarray() - tfidf_vectors[1].toarray())

    # Character n-gram overlap (n=3)
    # Reference: Used by [Jimenez et al., 2012], [Baer et al., 2012]
    char_ngrams1 = set([''.join(gram) for token in tokens1 for gram in ngrams(token, 3)])
    char_ngrams2 = set([''.join(gram) for token in tokens2 for gram in ngrams(token, 3)])
    char_overlap = len(char_ngrams1.intersection(char_ngrams2))
    char_union = len(char_ngrams1.union(char_ngrams2))
    char_ngram_overlap = char_overlap / char_union if char_union != 0 else 0

    # Character n-gram TF-IDF cosine similarity
    # Reference: Character-level features used by [Baer et al., 2012]
    char_tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4)).fit([s1, s2])
    char_tfidf_vectors = char_tfidf_vectorizer.transform([s1, s2])
    char_tfidf_cosine_sim = cosine_similarity(char_tfidf_vectors[0], char_tfidf_vectors[1])[0][0]

    # Word n-gram overlap (n=2)
    # Reference: Used by [Baer et al., 2012], [Štajner et al., 2012]
    ngram_overlap = ngram_overlap_ratio(tokens1_no_stop, tokens2_no_stop, n=2)

    # BLEU score
    # Reference: Used by [Baer et al., 2012]
    bleu_score = nltk.translate.bleu_score.sentence_bleu(
        [tokens1_no_stop], tokens2_no_stop,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)

    # Common words count
    # Reference: Basic text similarity
    common_word_count = len(overlap)

    # Total unique words count
    unique_word_count = len(union)

    # Ratio of common words to total unique words
    common_to_unique_ratio = common_word_count / unique_word_count if unique_word_count != 0 else 0

    # Length ratio
    # Reference: Used by [Baer et al., 2012]
    len_ratio = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) != 0 else 0

    # Absolute length difference
    # Reference: Additional lexical feature
    length_diff = abs(len(tokens1) - len(tokens2))

    # Content word overlap ratio (nouns, verbs, adjectives, adverbs)
    # Reference: Content word overlap used in similarity measures [Jimenez et al., 2012]
    pos_tags1_full = get_pos_tags(tokens1)
    pos_tags2_full = get_pos_tags(tokens2)
    content_pos_tags = ('NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                        'JJ', 'JJR', 'JJS',  # Adjectives
                        'RB', 'RBR', 'RBS')  # Adverbs
    tokens1_content = [w for w, pos in pos_tags1_full if pos in content_pos_tags]
    tokens2_content = [w for w, pos in pos_tags2_full if pos in content_pos_tags]

    content_overlap = set(tokens1_content).intersection(set(tokens2_content))
    content_union = set(tokens1_content).union(set(tokens2_content))
    content_word_overlap_ratio = len(content_overlap) / len(content_union) if len(content_union) != 0 else 0

    # Word n-gram precision and recall (n=1 to 3)
    ngram_precisions = []
    ngram_recalls = []
    for n in range(1, 4):
        ngrams1 = list(ngrams(tokens1_no_stop, n))
        ngrams2 = list(ngrams(tokens2_no_stop, n))
        ngrams1_set = set(ngrams1)
        ngrams2_set = set(ngrams2)
        overlap = ngrams1_set.intersection(ngrams2_set)
        total_ngrams1 = len(ngrams1)
        total_ngrams2 = len(ngrams2)
        precision = len(overlap) / total_ngrams2 if total_ngrams2 != 0 else 0
        recall = len(overlap) / total_ngrams1 if total_ngrams1 != 0 else 0
        ngram_precisions.append(precision)
        ngram_recalls.append(recall)

    # Longest Common Substring length normalized by average sentence length
    lcs_length = longest_common_substring_length(' '.join(tokens1), ' '.join(tokens2))
    avg_len = (len(' '.join(tokens1)) + len(' '.join(tokens2))) / 2
    lcs_norm = lcs_length / avg_len if avg_len != 0 else 0

    features = {
        'lex_jaccard': jaccard,
        'lex_dice_coeff': dice_coeff,
        'lex_overlap_coeff': overlap_coeff,
        'lex_norm_edit_distance': norm_edit_distance,
        'lex_jaro_winkler': jaro_winkler,
        'lex_cosine_sim': cosine_sim,
        'lex_euclidean_dist': euclidean_dist,
        'lex_char_ngram_overlap': char_ngram_overlap,
        'lex_char_tfidf_cosine_sim': char_tfidf_cosine_sim,
        'lex_ngram_overlap': ngram_overlap,
        'lex_bleu_score': bleu_score,
        'lex_common_word_count': common_word_count,
        'lex_common_to_unique_ratio': common_to_unique_ratio,
        'lex_len_ratio': len_ratio,
        'lex_length_diff': length_diff,
        'lex_content_word_overlap_ratio': content_word_overlap_ratio,
        'lex_ngram_precision_1': ngram_precisions[0],
        'lex_ngram_precision_2': ngram_precisions[1],
        'lex_ngram_precision_3': ngram_precisions[2],
        'lex_ngram_recall_1': ngram_recalls[0],
        'lex_ngram_recall_2': ngram_recalls[1],
        'lex_ngram_recall_3': ngram_recalls[2],
        'lex_lcs_norm': lcs_norm,
    }
    return features

def syntactic_features(s1, s2):
    """
    Compute syntactic similarity features between two sentences.

    Derived from methods in SemEval 2012 papers [Baer et al., 2012], [Štajner et al., 2012], [Glinos, 2012].

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of syntactic features.
    """
    tokens1 = preprocess_sentence(s1)
    tokens2 = preprocess_sentence(s2)

    pos_tags1_full = get_pos_tags(tokens1)
    pos_tags2_full = get_pos_tags(tokens2)

    pos_tags1 = [pos for _, pos in pos_tags1_full]
    pos_tags2 = [pos for _, pos in pos_tags2_full]

    # POS tag overlap ratio
    # Reference: Used by [Baer et al., 2012]
    pos_overlap = set(pos_tags1).intersection(set(pos_tags2))
    avg_pos_length = (len(pos_tags1) + len(pos_tags2)) / 2
    pos_overlap_ratio = len(pos_overlap) / avg_pos_length if avg_pos_length != 0 else 0

    # POS tag bigram overlap ratio
    # Reference: Used by [Baer et al., 2012]
    pos_bigram_overlap = ngram_overlap_ratio(pos_tags1, pos_tags2, n=2)

    # POS tag trigram overlap ratio
    # Reference: Used by [Baer et al., 2012]
    pos_trigram_overlap = ngram_overlap_ratio(pos_tags1, pos_tags2, n=3)

    # Dependency relation overlap
    # Reference: Used by [Štajner et al., 2012]
    deps1 = get_dependency_relations(s1)
    deps2 = get_dependency_relations(s2)
    dep_relations1 = set([dep[0] for dep in deps1])
    dep_relations2 = set([dep[0] for dep in deps2])
    dep_overlap = dep_relations1.intersection(dep_relations2)
    dep_union = dep_relations1.union(dep_relations2)
    dep_overlap_ratio = len(dep_overlap) / len(dep_union) if len(dep_union) != 0 else 0

    # Grammatical relations proportions
    # Reference: Used by [Baer et al., 2012]
    def compute_gram_rel_proportions(deps):
        total = len(deps)
        counts = {}
        for dep in deps:
            rel = dep[0]
            counts[rel] = counts.get(rel, 0) + 1
        proportions = {rel: count / total for rel, count in counts.items()}
        return proportions

    gram_rel_proportions1 = compute_gram_rel_proportions(deps1)
    gram_rel_proportions2 = compute_gram_rel_proportions(deps2)
    # Compute the cosine similarity between the grammatical relation proportions
    all_rels = set(list(gram_rel_proportions1.keys()) + list(gram_rel_proportions2.keys()))
    vector1 = [gram_rel_proportions1.get(rel, 0) for rel in all_rels]
    vector2 = [gram_rel_proportions2.get(rel, 0) for rel in all_rels]
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        gram_rel_cosine_sim = 0
    else:
        gram_rel_cosine_sim = cosine_similarity([vector1], [vector2])[0][0]

    # Word order similarity
    # Reference: Used by [Baer et al., 2012]
    word_order_sim = word_order_similarity(tokens1, tokens2)

    # Longest common subsequence
    # Reference: Used by [Baer et al., 2012], [Glinos, 2012]
    avg_length = (len(tokens1) + len(tokens2)) / 2
    lcs_length = longest_common_subsequence(tokens1, tokens2)
    lcs_norm = lcs_length / avg_length if avg_length != 0 else 0

    # Tree edit distance (simplified)
    # Reference: Used by [Štajner et al., 2012]
    tree_edit_dist = tree_edit_distance(s1, s2)

    # POS tag sequence similarity (normalized edit distance)
    # Reference: Used by [Glinos, 2012]
    pos_edit_distance = nltk.edit_distance(pos_tags1, pos_tags2)
    max_len = max(len(pos_tags1), len(pos_tags2))
    pos_norm_edit_distance = 1 - (pos_edit_distance / max_len) if max_len != 0 else 0

    # POS n-gram precision and recall (n=1 to 2)
    pos_ngram_precisions = []
    pos_ngram_recalls = []
    for n in range(1, 3):
        pos_ngrams1 = list(ngrams(pos_tags1, n))
        pos_ngrams2 = list(ngrams(pos_tags2, n))
        pos_ngrams1_set = set(pos_ngrams1)
        pos_ngrams2_set = set(pos_ngrams2)
        overlap = pos_ngrams1_set.intersection(pos_ngrams2_set)
        total_ngrams1 = len(pos_ngrams1)
        total_ngrams2 = len(pos_ngrams2)
        precision = len(overlap) / total_ngrams2 if total_ngrams2 != 0 else 0
        recall = len(overlap) / total_ngrams1 if total_ngrams1 != 0 else 0
        pos_ngram_precisions.append(precision)
        pos_ngram_recalls.append(recall)

    # Dependency relation precision and recall
    dep_overlap_all = set(deps1).intersection(set(deps2))
    dep_precision = len(dep_overlap_all) / len(deps2) if len(deps2) != 0 else 0
    dep_recall = len(dep_overlap_all) / len(deps1) if len(deps1) != 0 else 0

    # POS tag counts difference
    # Reference: Considering POS tag distributions as features, inspired by [Glinos, 2012]
    def pos_tag_counts(pos_tags):
        counts = {'N': 0, 'V': 0, 'J': 0, 'R': 0}
        for tag in pos_tags:
            if tag.startswith('N'):
                counts['N'] += 1
            elif tag.startswith('V'):
                counts['V'] += 1
            elif tag.startswith('J'):
                counts['J'] += 1
            elif tag.startswith('R'):
                counts['R'] += 1
        return counts

    counts1 = pos_tag_counts(pos_tags1)
    counts2 = pos_tag_counts(pos_tags2)

    pos_diff_features = {
        'pos_noun_diff': abs(counts1['N'] - counts2['N']),
        'pos_verb_diff': abs(counts1['V'] - counts2['V']),
        'pos_adj_diff': abs(counts1['J'] - counts2['J']),
        'pos_adv_diff': abs(counts1['R'] - counts2['R']),
    }

    features = {
        'syn_pos_overlap_ratio': pos_overlap_ratio,
        'syn_pos_bigram_overlap': pos_bigram_overlap,
        'syn_pos_trigram_overlap': pos_trigram_overlap,
        'syn_dep_overlap_ratio': dep_overlap_ratio,
        'syn_gram_rel_cosine_sim': gram_rel_cosine_sim,
        'syn_word_order_sim': word_order_sim,
        'syn_lcs_norm': lcs_norm,
        'syn_tree_edit_dist': tree_edit_dist,
        'syn_pos_norm_edit_distance': pos_norm_edit_distance,
        'syn_pos_ngram_precision_1': pos_ngram_precisions[0],
        'syn_pos_ngram_precision_2': pos_ngram_precisions[1],
        'syn_pos_ngram_recall_1': pos_ngram_recalls[0],
        'syn_pos_ngram_recall_2': pos_ngram_recalls[1],
        'syn_dep_precision': dep_precision,
        'syn_dep_recall': dep_recall,
    }
    features.update(pos_diff_features)
    return features

def semantic_features(s1, s2):
    """
    Compute semantic similarity features between two sentences.

    Features inspired by methods used in SemEval 2012 Task 6 participant papers.

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of semantic features.
    """
    tokens1 = preprocess_sentence(s1)
    tokens2 = preprocess_sentence(s2)

    # Synonym Matching Based on WordNet Synsets
    # Reference: Used by [Baer et al., 2012], [Jimenez et al., 2012]
    synonym_overlap = synonym_overlap_ratio(tokens1, tokens2)

    # Lexical Chain Overlap
    # Reference: Used by [Baer et al., 2012]
    lexical_chain_overlap = lexical_chain_overlap_ratio(tokens1, tokens2)

    # WordNet-based similarity measures
    # Reference: Used by the UKP team [Baer et al., 2012] and TakeLab team [Štajner et al., 2012]

    # Define similarity functions for WordNet
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
                                total_sim += sim
                                count += 1

        avg_sim = total_sim / count if count > 0 else 0
        return avg_sim, max_sim


    # WordNet Path Similarity (Average and Max)
    # Reference: Used by [Baer et al., 2012]
    avg_sim_path, max_sim_path = avg_max_wordnet_similarity(tokens1, tokens2, lambda s1, s2: s1.path_similarity(s2))

    # WordNet Wu-Palmer Similarity (Average and Max)
    # Reference: Used by [Baer et al., 2012], [Štajner et al., 2012]
    avg_sim_wup, max_sim_wup = avg_max_wordnet_similarity(tokens1, tokens2, lambda s1, s2: s1.wup_similarity(s2))

    # WordNet Leacock-Chodorow Similarity (Average and Max)
    # Reference: Used by [Baer et al., 2012]
    avg_sim_lch, max_sim_lch = avg_max_wordnet_similarity(tokens1, tokens2, lambda s1, s2: s1.lch_similarity(s2))

    # Antonym overlap ratio
    # Reference: Capturing antonyms and negation differences, inspired by [Baer et al., 2012]
    antonym_ratio = antonym_overlap(tokens1, tokens2)

    # Named entity overlap (enhanced)
    # Reference: Used by [Baer et al., 2012]
    ne_overlap = named_entity_overlap(s1, s2)
    ne_type_overlap = named_entity_type_overlap(s1, s2)

    # Simplified Lesk-based similarity
    # Reference: Used by [Baer et al., 2012]
    simplified_lesk_sim = simplified_lesk_similarity(tokens1, tokens2)

    # Hypernym/Hyponym Overlap
    # Reference: Used by [Baer et al., 2012], [Štajner et al., 2012]
    hypernym_hyponym_overlap = hypernym_hyponym_overlap_ratio(tokens1, tokens2)

    # LSA and LDA similarities
    # Reference: Used by [Baer et al., 2012]
    lsa_sim = lsa_similarity(s1, s2)
    lda_sim = lda_similarity(s1, s2)

    # Sentiment scores and differences using SentiWordNet
    # Reference: Incorporating sentiment features as in [Gupta et al., 2012]
    pos1, neg1, obj1 = compute_sentiment_score(tokens1)
    pos2, neg2, obj2 = compute_sentiment_score(tokens2)

    sentiment_diff = {
        'sem_sentiment_pos_diff': abs(pos1 - pos2),
        'sem_sentiment_neg_diff': abs(neg1 - neg2),
        'sem_sentiment_obj_diff': abs(obj1 - obj2),
    }

    # Negation features
    # Reference: Considering negation as an important factor, as in [Baer et al., 2012]
    neg_count1 = count_negations(tokens1)
    neg_count2 = count_negations(tokens2)
    negation_feature = {
        'sem_negation_difference': abs(neg_count1 - neg_count2),
        'sem_negation_both_present': int(neg_count1 > 0 and neg_count2 > 0),
        'sem_negation_both_absent': int(neg_count1 == 0 and neg_count2 == 0)
    }

    features = {
        'sem_synonym_overlap': synonym_overlap,
        'sem_lexical_chain_overlap': lexical_chain_overlap,
        'sem_ne_overlap': ne_overlap,
        'sem_ne_type_overlap': ne_type_overlap,
        'sem_simplified_lesk_sim': simplified_lesk_sim,
        'sem_hypernym_hyponym_overlap': hypernym_hyponym_overlap,
        'sem_lsa_sim': lsa_sim,
        'sem_lda_sim': lda_sim,
        'sem_avg_sim_path': avg_sim_path,
        'sem_max_sim_path': max_sim_path,
        'sem_avg_sim_wup': avg_sim_wup,
        'sem_max_sim_wup': max_sim_wup,
        'sem_avg_sim_lch': avg_sim_lch,
        'sem_max_sim_lch': max_sim_lch,
        'sem_antonym_ratio': antonym_ratio,
    }
    features.update(sentiment_diff)
    features.update(negation_feature)
    return features

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

def named_entity_overlap(s1, s2):
    """
    Calculate named entity overlap between two sentences.
    """
    doc1 = nlp(s1)
    doc2 = nlp(s2)

    ne1 = set([ent.text.lower() for ent in doc1.ents])
    ne2 = set([ent.text.lower() for ent in doc2.ents])

    overlap = ne1.intersection(ne2)
    union = ne1.union(ne2)
    return len(overlap) / len(union) if len(union) else 0

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

    if n_features >= 2:
        n_components = min(100, n_features - 1)
        svd = TruncatedSVD(n_components=n_components)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        lsa_matrix = lsa.fit_transform(tfidf_matrix)
        sim = cosine_similarity([lsa_matrix[0]], [lsa_matrix[1]])[0][0]
        if np.isnan(sim):
            sim = 0.0
    else:
        # Fallback to cosine similarity of TF-IDF vectors
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
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

def antonym_overlap(tokens1, tokens2):
    """
    Compute the ratio of antonym pairs between two lists of tokens.

    Reference: Inspired by [Baer et al., 2012]

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
