# feature_extraction/lexical_features.py

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
from nltk.metrics.distance import jaro_winkler_similarity
from scipy.stats import pearsonr

from feature_extraction.feature_utils import (
    preprocess_sentence,
    get_pos_tags,
    ngram_overlap_ratio,
    greedy_string_tiling,
    longest_common_substring_length,
    soft_jaccard_similarity,
)


def lexical_features(s1, s2):
    """
    Compute lexical similarity features between two sentences.

    Derived from methods used in SemEval 2012 papers.

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of lexical features.
    """
    features = {}
    tokens1 = preprocess_sentence(s1)
    tokens2 = preprocess_sentence(s2)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens1_no_stop = [w for w in tokens1 if w not in stopwords]
    tokens2_no_stop = [w for w in tokens2 if w not in stopwords]

    # Word overlap
    overlap = set(tokens1_no_stop).intersection(set(tokens2_no_stop))
    union = set(tokens1_no_stop).union(set(tokens2_no_stop))
    word_overlap_ratio = len(overlap) / len(union) if len(union) != 0 else 0

    # Jaccard similarity
    jaccard = word_overlap_ratio

    # Dice coefficient
    dice_coeff = (2 * len(overlap)) / (len(tokens1_no_stop) + len(tokens2_no_stop)) \
        if (len(tokens1_no_stop) + len(tokens2_no_stop)) != 0 else 0

    # Overlap coefficient
    min_len = min(len(tokens1_no_stop), len(tokens2_no_stop))
    overlap_coeff = len(overlap) / min_len if min_len != 0 else 0

    # Levenshtein Distance (Edit Distance)
    edit_distance = nltk.edit_distance(' '.join(tokens1), ' '.join(tokens2))
    max_len = max(len(' '.join(tokens1)), len(' '.join(tokens2)))
    norm_edit_distance = 1 - (edit_distance / max_len) if max_len != 0 else 0

    # Jaro-Winkler similarity
    jaro_winkler = jaro_winkler_similarity(' '.join(tokens1), ' '.join(tokens2))

    # Cosine similarity using TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer().fit([s1, s2])
    tfidf_vectors = tfidf_vectorizer.transform([s1, s2])
    cosine_sim = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]

    # Euclidean distance between TF-IDF vectors
    euclidean_dist = np.linalg.norm(tfidf_vectors[0].toarray() - tfidf_vectors[1].toarray())

    # Character n-gram overlaps (n=2,3,4)
    char_ngram_overlaps = {}
    for n in [2, 3, 4]:
        char_ngrams1 = set([''.join(gram) for token in tokens1 for gram in ngrams(token, n)])
        char_ngrams2 = set([''.join(gram) for token in tokens2 for gram in ngrams(token, n)])
        char_overlap = len(char_ngrams1.intersection(char_ngrams2))
        char_union = len(char_ngrams1.union(char_ngrams2))
        char_ngram_overlap = char_overlap / char_union if char_union != 0 else 0
        char_ngram_overlaps[f'lex_char_{n}gram_overlap'] = char_ngram_overlap

    # Character n-gram TF-IDF cosine similarity
    char_tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4)).fit([s1, s2])
    char_tfidf_vectors = char_tfidf_vectorizer.transform([s1, s2])
    char_tfidf_cosine_sim = cosine_similarity(
        char_tfidf_vectors[0], char_tfidf_vectors[1]
    )[0][0]

    # Word n-gram overlap (n=2)
    ngram_overlap = ngram_overlap_ratio(tokens1_no_stop, tokens2_no_stop, n=2)

    # Word n-gram overlap using Containment measure (n=1,2)
    word_ngram_containments = {}
    for n in [1, 2]:
        ngrams1 = set(ngrams(tokens1_no_stop, n))
        ngrams2 = set(ngrams(tokens2_no_stop, n))
        intersection = ngrams1.intersection(ngrams2)
        containment = len(intersection) / min(len(ngrams1), len(ngrams2)) \
            if min(len(ngrams1), len(ngrams2)) != 0 else 0
        word_ngram_containments[f'lex_word_{n}gram_containment'] = containment

    # Word n-gram overlap using Jaccard coefficient (n=1,3,4), with and without stopwords
    word_ngram_jaccards = {}
    for n in [1, 3, 4]:
        for stopword_setting, tokens1_set, tokens2_set in [
            ('with_stop', tokens1, tokens2), ('no_stop', tokens1_no_stop, tokens2_no_stop)
        ]:
            ngrams1 = set(ngrams(tokens1_set, n))
            ngrams2 = set(ngrams(tokens2_set, n))
            intersection = ngrams1.intersection(ngrams2)
            union = ngrams1.union(ngrams2)
            jaccard = len(intersection) / len(union) if len(union) != 0 else 0
            word_ngram_jaccards[
                f'lex_word_{n}gram_jaccard_{stopword_setting}'
            ] = jaccard

    # Stopword n-gram overlaps (n=2 to 10)
    stopword_tokens1 = [w for w in tokens1 if w in stopwords]
    stopword_tokens2 = [w for w in tokens2 if w in stopwords]

    stopword_ngram_overlaps = {}
    for n in range(2, 11):
        ngrams1 = set(ngrams(stopword_tokens1, n))
        ngrams2 = set(ngrams(stopword_tokens2, n))
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        overlap_aux = len(intersection) / len(union) if len(union) != 0 else 0
        stopword_ngram_overlaps[f'lex_stopword_{n}gram_overlap'] = overlap_aux

    # Function word frequencies
    function_words = set([
        'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'was', 'he', 'for', 'it',
        'with', 'as', 'his', 'on', 'be', 'at', 'by', 'i', 'this', 'had', 'not',
        'are', 'but', 'from', 'or', 'have', 'an', 'they', 'which', 'one', 'you',
        'were', 'her', 'all', 'she', 'there', 'would', 'their', 'we', 'him', 'been',
        'has', 'when', 'who', 'will', 'no', 'more', 'if', 'out', 'so', 'said',
        'what', 'up', 'its', 'about', 'than', 'into', 'them', 'can', 'only', 'other',
        'new', 'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first',
        'any', 'my', 'now', 'such', 'like', 'our', 'over', 'man', 'me', 'even',
        'most', 'made', 'after', 'also', 'did', 'many', 'before', 'must', 'through',
        'back', 'years', 'where', 'much', 'your', 'way', 'well', 'down', 'should',
        'because', 'each', 'just', 'those', 'people'
    ])

    def get_function_word_frequencies(tokens, function_words):
        freq_dict = {word: 0 for word in function_words}
        total = 0
        for token in tokens:
            if token in freq_dict:
                freq_dict[token] += 1
                total += 1
        if total > 0:
            for word in freq_dict:
                freq_dict[word] /= total
        return freq_dict

    freqs1 = get_function_word_frequencies(tokens1, function_words)
    freqs2 = get_function_word_frequencies(tokens2, function_words)

    function_words_common = [
        word for word in function_words if freqs1[word] > 0 or freqs2[word] > 0
    ]
    if len(function_words_common) >= 2:
        values1 = [freqs1[word] for word in function_words_common]
        values2 = [freqs2[word] for word in function_words_common]

        # Check for constant arrays
        if len(set(values1)) == 1 or len(set(values2)) == 1:
            function_word_corr = 0  # Correlation is undefined for constant arrays
        else:
            correlation, _ = pearsonr(values1, values2)
            function_word_corr = correlation if not np.isnan(correlation) else 0
    else:
        function_word_corr = 0

    # Greedy String Tiling similarity
    gst_similarity = greedy_string_tiling(
        ' '.join(tokens1), ' '.join(tokens2), min_match_length=3
    )

    # Longest Common Substring length normalized by average sentence length
    lcs_length = longest_common_substring_length(
        ' '.join(tokens1), ' '.join(tokens2)
    )
    avg_len = (len(' '.join(tokens1)) + len(' '.join(tokens2))) / 2
    lcs_norm = lcs_length / avg_len if avg_len != 0 else 0

    # BLEU score
    bleu_score = nltk.translate.bleu_score.sentence_bleu(
        [tokens1_no_stop], tokens2_no_stop,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
    )

    # Common words count
    common_word_count = len(overlap)

    # Total unique words count
    unique_word_count = len(union)

    # Ratio of common words to total unique words
    common_to_unique_ratio = common_word_count / unique_word_count \
        if unique_word_count != 0 else 0

    # Length ratio
    len_ratio = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) \
        if max(len(tokens1), len(tokens2)) != 0 else 0

    # Absolute length difference
    length_diff = abs(len(tokens1) - len(tokens2))

    # Content word overlap ratio (nouns, verbs, adjectives, adverbs)
    pos_tags1_full = get_pos_tags(tokens1)
    pos_tags2_full = get_pos_tags(tokens2)
    content_pos_tags = (
        'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
        'JJ', 'JJR', 'JJS',  # Adjectives
        'RB', 'RBR', 'RBS'   # Adverbs
    )
    tokens1_content = [w for w, pos in pos_tags1_full if pos in content_pos_tags]
    tokens2_content = [w for w, pos in pos_tags2_full if pos in content_pos_tags]

    content_overlap = set(tokens1_content).intersection(set(tokens2_content))
    content_union = set(tokens1_content).union(set(tokens2_content))
    content_word_overlap_ratio = len(content_overlap) / len(content_union) \
        if len(content_union) != 0 else 0

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

    # Soft Cardinality Similarity (Lexical)
    tokens1_no_stop_set = set(tokens1_no_stop)
    tokens2_no_stop_set = set(tokens2_no_stop)
    soft_jaccard = soft_jaccard_similarity(tokens1_no_stop_set, tokens2_no_stop_set)

    # Type-Token Ratio (TTR)
    ttr1 = len(set(tokens1)) / len(tokens1) if len(tokens1) > 0 else 0
    ttr2 = len(set(tokens2)) / len(tokens2) if len(tokens2) > 0 else 0
    ttr_diff = abs(ttr1 - ttr2)

    features = {
        'lex_jaccard': jaccard,
        'lex_dice_coeff': dice_coeff,
        'lex_overlap_coeff': overlap_coeff,
        'lex_norm_edit_distance': norm_edit_distance,
        'lex_jaro_winkler': jaro_winkler,
        'lex_cosine_sim': cosine_sim,
        'lex_euclidean_dist': euclidean_dist,
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
        'lex_soft_jaccard': soft_jaccard,
        'lex_function_word_corr': function_word_corr,
        'lex_gst_similarity': gst_similarity,
        'lex_ttr_diff': ttr_diff,
    }
    features.update(char_ngram_overlaps)
    features.update(word_ngram_containments)
    features.update(word_ngram_jaccards)
    features.update(stopword_ngram_overlaps)
    return features
