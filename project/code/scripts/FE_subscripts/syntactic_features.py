from nltk.util import ngrams

import spacy
nlp = spacy.load('en_core_web_sm')


from .feature_utils import (
    jaccard_similarity,
    containment_similarity,
    harmonic_mean,
    wordnet_similarity
)
# ---------------------------- #
# SYNTACTIC FEATURE EXTRACTION #
# ---------------------------- #

def syntactic_features_extractor(s1_proc, s2_proc):
    """
    Syntactic Features include:
    - Structural similarity measures (from UKP):
        - Stopword n-grams
        - POS n-grams
        - Word Pair Order
        - Word Pair Distance
    - Syntactic features (from TakeLab):
        - Syntactic Roles Similarity
        - Syntactic Dependencies Overlap
    """
    features = {}

    # --- UKP Syntactic Features ---

    # Stopword n-grams (n=2 to 10) (UKP)
    stop_tokens1 = s1_proc['stopwords']
    stop_tokens2 = s2_proc['stopwords']
    for n in range(2, 11):
        ngrams1 = set(ngrams(stop_tokens1, n))
        ngrams2 = set(ngrams(stop_tokens2, n))
        containment = containment_similarity(ngrams1, ngrams2)
        features[f'stopword_ngram_overlap_{n}'] = containment

    # POS n-grams (UKP)
    pos_tags1 = [pos for (_, pos) in s1_proc['pos_tagged_tokens']]
    pos_tags2 = [pos for (_, pos) in s2_proc['pos_tagged_tokens']]
    for n in [2, 3, 4]:
        ngrams1 = set(ngrams(pos_tags1, n))
        ngrams2 = set(ngrams(pos_tags2, n))
        jaccard = jaccard_similarity(ngrams1, ngrams2)
        features[f'pos_{n}gram_jaccard'] = jaccard

    # Word Pair Order (UKP) - considering all pairs in order
    wpo_score = word_pair_order(s1_proc['lemmas'], s2_proc['lemmas'])
    features['word_pair_order'] = wpo_score

    # Word Pair Distance (UKP)
    wpd_score = word_pair_distance(s1_proc['lemmas'], s2_proc['lemmas'])
    features['word_pair_distance'] = wpd_score

    # --- TakeLab Syntactic Features ---

    # Parse sentences with spaCy
    doc1 = nlp(s1_proc['text'])
    doc2 = nlp(s2_proc['text'])

    # Syntactic Roles Similarity (TakeLab)
    roles1 = extract_syntactic_roles(doc1)
    roles2 = extract_syntactic_roles(doc2)

    # Compute similarities for each role
    for role in ['subject', 'object', 'predicate']:
        words1 = roles1.get(role, [])
        words2 = roles2.get(role, [])
        max_sim = 0.0
        for w1 in words1:
            for w2 in words2:
                sim = wordnet_similarity(w1.lemma_, w2.lemma_)
                if sim and sim > max_sim:
                    max_sim = sim
        features[f'syntactic_similarity_{role}'] = max_sim

    # Syntactic Dependencies Overlap (TakeLab)
    deps1 = set([(token.dep_, token.head.lemma_, token.lemma_) for token in doc1])
    deps2 = set([(token.dep_, token.head.lemma_, token.lemma_) for token in doc2])

    dep_overlap = harmonic_mean(
        len(deps1 & deps2) / len(deps1) if deps1 else 0.0,
        len(deps1 & deps2) / len(deps2) if deps2 else 0.0
    )
    features['dependency_overlap'] = dep_overlap

    features = {f'syn_{key}': value for key, value in features.items()}

    return features

# --------------------------- #
#           FUNCTIONS         #
# --------------------------- #


def word_pair_order(lemmas1, lemmas2):
    """
    Compute word pair order similarity considering all pairs in order.

    Args:
        lemmas1 (list): List of lemmas from s1.
        lemmas2 (list): List of lemmas from s2.

    Returns:
        float: Word pair order similarity.
    """
    pairs1 = [(lemmas1[i], lemmas1[j]) for i in range(len(lemmas1)) for j in range(i+1, len(lemmas1))]
    pairs2 = [(lemmas2[i], lemmas2[j]) for i in range(len(lemmas2)) for j in range(i+1, len(lemmas2))]
    matches = sum(1 for pair in pairs1 if pair in pairs2)
    total_pairs = min(len(pairs1), len(pairs2))
    return matches / total_pairs if total_pairs > 0 else 0.0

def word_pair_distance(lemmas1, lemmas2):
    """
    Compute word pair distance similarity.

    Args:
        lemmas1 (list): List of lemmas from s1.
        lemmas2 (list): List of lemmas from s2.

    Returns:
        float: Word pair distance similarity.
    """
    distance_sum = 0.0
    count = 0
    for w1 in lemmas1:
        if w1 in lemmas2:
            i = lemmas1.index(w1)
            j = lemmas2.index(w1)
            distance = abs(i - j)
            distance_sum += distance
            count += 1
    avg_distance = distance_sum / count if count > 0 else float('inf')
    return 1 / (1 + avg_distance) if avg_distance != float('inf') else 0.0

def extract_syntactic_roles(doc):
    """
    Extract syntactic roles from a spaCy Doc object.

    Args:
        doc: spaCy Doc object.

    Returns:
        dict: A dictionary with roles as keys and lists of tokens as values.
    """
    roles = {'subject': [], 'object': [], 'predicate': []}
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass'):
            roles['subject'].append(token)
        elif token.dep_ in ('dobj', 'pobj', 'iobj'):
            roles['object'].append(token)
        elif token.pos_ == 'VERB':
            roles['predicate'].append(token)
    return roles