import numpy as np
from scipy.stats import pearsonr


# ---------------------------- #
# STYLE FEATURE EXTRACTION     #
# ---------------------------- #

def style_features_extractor(s1_proc, s2_proc):
    """
    Style Features include:
    - Function Word Frequencies (from UKP)
    - Type-Token Ratio (from UKP)
    - Sequential Type-Token Ratio (from UKP)
    """
    features = {}

    # Function Word Frequencies (UKP)
    func_word_freq_similarity = function_word_frequencies(
        s1_proc['function_words'], s2_proc['function_words']
        )
    features['function_word_freq_similarity'] = func_word_freq_similarity

    # Type-Token Ratio (UKP)
    ttr_s1 = type_token_ratio(s1_proc['tokens_lower'])
    ttr_s2 = type_token_ratio(s2_proc['tokens_lower'])
    ttr_diff = abs(ttr_s1 - ttr_s2)
    features['ttr_difference'] = ttr_diff

    # Sequential Type-Token Ratio (UKP)
    sttr_s1 = sequential_ttr(s1_proc['tokens_lower'])
    sttr_s2 = sequential_ttr(s2_proc['tokens_lower'])
    sttr_diff = abs(sttr_s1 - sttr_s2)
    features['sttr_difference'] = sttr_diff

    features = {f'sty_{key}': value for key, value in features.items()}

    return features


# --------------------------- #
#           FUNCTIONS         #
# --------------------------- #

def function_word_frequencies(fw1, fw2):
    """
    Compute the Pearson correlation of function word frequencies.

    Args:
        fw1 (list): List of function words in s1.
        fw2 (list): List of function words in s2.

    Returns:
        float: Pearson correlation coefficient.
    """
    
    # unique function words from both lists
    all_fw = set(fw1 + fw2)
    
    freq1 = [fw1.count(w) for w in all_fw]
    freq2 = [fw2.count(w) for w in all_fw]
    
    # if insuficient or constant
    if len(freq1) < 2 or len(freq2) < 2:
        return 0.0
    
    if np.var(freq1) == 0 or np.var(freq2) == 0:
        return 0.0
    
    corr, _ = pearsonr(freq1, freq2)
    return corr

def type_token_ratio(tokens):
    """
    Compute the type-token ratio.

    Args:
        tokens (list): List of tokens.

    Returns:
        float: Type-token ratio.
    """
    types = set(tokens)
    return len(types) / len(tokens) if tokens else 0.0

def sequential_ttr(tokens):
    """
    Compute sequential TTR as average TTR over sequences.

    Args:
        tokens (list): List of tokens.

    Returns:
        float: Sequential TTR.
    """
    window_size = 20
    ttrs = []
    for i in range(0, len(tokens), window_size):
        window = tokens[i:i + window_size]
        ttr = type_token_ratio(window)
        ttrs.append(ttr)
    return np.mean(ttrs) if ttrs else 0.0