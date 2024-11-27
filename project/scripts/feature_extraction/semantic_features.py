# feature_extraction/semantic_features.py

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from feature_extraction.feature_utils import (
    preprocess_sentence,
    synonym_overlap_ratio,
    lexical_chain_overlap_ratio,
    avg_max_wordnet_similarity,
    antonym_overlap,
    named_entity_overlap,
    named_entity_type_overlap,
    simplified_lesk_similarity,
    hypernym_hyponym_overlap_ratio,
    lsa_similarity,
    lda_similarity,
    compute_sentiment_score,
    count_negations,
    semantic_role_overlap,
    temporal_expression_overlap,
)


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
    features = {}
    tokens1 = preprocess_sentence(s1)
    tokens2 = preprocess_sentence(s2)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens1_no_stop = [w for w in tokens1 if w not in stopwords]
    tokens2_no_stop = [w for w in tokens2 if w not in stopwords]

    # Synonym Matching Based on WordNet Synsets
    synonym_overlap = synonym_overlap_ratio(tokens1, tokens2)

    # Lexical Chain Overlap
    lexical_chain_overlap = lexical_chain_overlap_ratio(tokens1, tokens2)

    # WordNet-based similarity measures
    # WordNet Path Similarity (Average and Max)
    avg_sim_path, max_sim_path = avg_max_wordnet_similarity(
        tokens1, tokens2, lambda s1, s2: s1.path_similarity(s2)
    )

    # WordNet Wu-Palmer Similarity (Average and Max)
    avg_sim_wup, max_sim_wup = avg_max_wordnet_similarity(
        tokens1, tokens2, lambda s1, s2: s1.wup_similarity(s2)
    )

    # WordNet Leacock-Chodorow Similarity (Average and Max)
    avg_sim_lch, max_sim_lch = avg_max_wordnet_similarity(
        tokens1, tokens2, lambda s1, s2: s1.lch_similarity(s2)
    )

    # Antonym overlap ratio
    antonym_ratio = antonym_overlap(tokens1, tokens2)

    # Named entity overlap
    ne_overlap = named_entity_overlap(s1, s2)
    ne_type_overlap = named_entity_type_overlap(s1, s2)

    # Simplified Lesk-based similarity
    simplified_lesk_sim = simplified_lesk_similarity(tokens1, tokens2)

    # Hypernym/Hyponym Overlap
    hypernym_hyponym_overlap = hypernym_hyponym_overlap_ratio(tokens1, tokens2)

    # LSA and LDA similarities
    lsa_sim = lsa_similarity(s1, s2)
    lda_sim = lda_similarity(s1, s2)

    # Sentiment scores and differences using SentiWordNet
    pos1, neg1, obj1 = compute_sentiment_score(tokens1)
    pos2, neg2, obj2 = compute_sentiment_score(tokens2)

    sentiment_diff = {
        'sem_sentiment_pos_diff': abs(pos1 - pos2),
        'sem_sentiment_neg_diff': abs(neg1 - neg2),
        'sem_sentiment_obj_diff': abs(obj1 - obj2),
    }

    # Negation features
    neg_count1 = count_negations(tokens1)
    neg_count2 = count_negations(tokens2)
    negation_feature = {
        'sem_negation_difference': abs(neg_count1 - neg_count2),
        'sem_negation_both_present': int(neg_count1 > 0 and neg_count2 > 0),
        'sem_negation_both_absent': int(neg_count1 == 0 and neg_count2 == 0)
    }

    # Semantic Role Labeling (SRL) Overlap
    sem_srl_overlap = semantic_role_overlap(s1, s2)

    # Temporal Expression Overlap
    sem_temporal_overlap = temporal_expression_overlap(s1, s2)

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
        'sem_srl_overlap': sem_srl_overlap,
        'sem_temporal_overlap': sem_temporal_overlap,
    }
    features.update(sentiment_diff)
    features.update(negation_feature)
    return features
