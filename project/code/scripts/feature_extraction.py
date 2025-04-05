# feature_extraction.py

import functools
import warnings

from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import brown, wordnet as wn, wordnet_ic


from .FE_subscripts.lexical_features import lexical_features_extractor
from .FE_subscripts.syntactic_features import syntactic_features_extractor
from .FE_subscripts.semantic_features import semantic_features_extractor
from .FE_subscripts.stylistic_features import style_features_extractor
from .FE_subscripts.submodels import (
    ESA_Model,
    LexSubModel,
    SMT_Model,
    DistributionalThesaurus
)


warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning,
    module="sklearn.decomposition._truncated_svd"
    )


# needed nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet_ic', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------------------------- #
#  FEATURE EXTRACTOR FUNCTION  #
# ---------------------------- #

class FeatureExtractor:
    def __init__(self):
        """Initialize the FeatureExtractor with an internal memoization dictionary."""
        self.memoized_pair_features = {}

        self.esa_model      = ESA_Model()
        self.lexsub_model   = LexSubModel()
        self.smt_model      = SMT_Model()
        self.dist_thesaurus = DistributionalThesaurus()

    def extract_pair_features(self, s1, s2):
        """Extract features for a pair of sentences with order-invariant memoization."""
        # we use frozenset to make memoization invariant to the order of s1 and s2
        key = frozenset([s1, s2])
        if key in self.memoized_pair_features:
            return self.memoized_pair_features[key]
        features = extract_features(s1, s2, self.esa_model, self.lexsub_model, self.smt_model, self.dist_thesaurus)
        self.memoized_pair_features[key] = features
        return features

    def process_row(self, row):
        """Process a single row and extract features for the sentence pair."""
        s1, s2, score, ds = row

        features = self.extract_pair_features(s1, s2)

        features['score'] = score
        features['dataset'] = ds

        return features

    def extract_features_parallel(self, data):
        """Extract features in parallel and return a DataFrame."""
        with Pool(cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(self.process_row, data),
                    total=len(data),
                    desc="Extracting Features"
                )
            )
        return pd.DataFrame(results)

    def extract_features_sequential(self, data):
        """Extract features sequentially and return a DataFrame."""
        results = []
        for row in tqdm(data, desc="Extracting Features (Sequential)"):
            results.append(self.process_row(row))
        return pd.DataFrame(results)

# ----------------------- #
# ALL RELEVANT FUNCTIONS  #
# ----------------------- #

@functools.lru_cache(maxsize=None)
def extract_features(s1, s2, esa_model, lexsub_model, smt_model, dist_thesaurus):
    """
    Extracts lexical, syntactic, and semantic features.

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.
        esa_model: An instance of ESA_Model for computing ESA similarity.
        lexsub_model: An instance of LexSubModel for lexical substitution.
        smt_model: An instance of SMT_Model for back-translation.
        dist_thesaurus: An instance of DistributionalThesaurus.

    Returns:
        dict: A dictionary of combined features.
    """
    features = {}

    s1_proc = preprocess(s1)
    s2_proc = preprocess(s2)

    features.update(lexical_features_extractor(s1_proc, s2_proc))
    features.update(syntactic_features_extractor(s1_proc, s2_proc))
    features.update(semantic_features_extractor(s1_proc, s2_proc, esa_model, lexsub_model, smt_model, dist_thesaurus))
    features.update(style_features_extractor(s1_proc, s2_proc))

    return features



# ---------------------------- #
# HELPER FUNCTIONS             #
# ---------------------------- #

def preprocess(s):
    """
    Preprocess a sentence.

    Args:
        s (str): Input sentence.

    Returns:
        dict: A dictionary with tokens, lemmas, stopwords removed tokens, etc.
    """
    # original text
    text = s

    # text without spaces (for character-based measures)
    text_no_space = ''.join(s.split())

    # tokenize sentence
    tokens = nltk.word_tokenize(s)
    tokens_lower = [token.lower() for token in tokens]
    tokens_lower_nopunct = [token.lower() for token in tokens if token.isalpha()]
    tokens_no_stop = [token.lower() for token in tokens if token.lower() not in stopwords.words('english')]

    # POS tagging
    pos_tagged_tokens = nltk.pos_tag(tokens)
    
    # lemmas
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    content_tokens = []
    content_lemmas = []
    function_words = []
    stop_words = set(stopwords.words('english'))
    stopwords_tokens = []
    numbers = []

    for token, pos in pos_tagged_tokens:
        pos_wn = get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(token.lower(), pos=pos_wn) if pos_wn else token.lower()
        lemmas.append(lemma)
        if token.lower() in stop_words:
            function_words.append(token.lower())
            stopwords_tokens.append(token.lower())
        else:
            if pos.startswith(('J', 'V', 'N', 'R')):
                content_tokens.append(token.lower())
                content_lemmas.append(lemma)

        if token.isdigit():
            numbers.append(token)

    return {
        'text': text,
        'text_no_space': text_no_space,
        'tokens': tokens,
        'tokens_lower': tokens_lower,
        'tokens_lower_nopunct': tokens_lower_nopunct,
        'tokens_no_stop': tokens_no_stop,
        'pos_tagged_tokens': pos_tagged_tokens,
        'lemmas': lemmas,
        'lemmas_no_stop': [lemma for lemma in lemmas if lemma not in stop_words],
        'content_tokens': content_tokens,
        'content_lemmas': content_lemmas,
        'stopwords': stopwords_tokens,
        'function_words': function_words,
        'numbers': numbers
    }

def get_wordnet_pos(treebank_tag):
    """
    Map POS tag to first character lemmatize() accepts.

    Args:
        treebank_tag (str): POS tag.

    Returns:
        char or None: Corresponding WordNet tag or None.
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    if treebank_tag.startswith('V'):
        return wn.VERB
    if treebank_tag.startswith('N'):
        return wn.NOUN
    if treebank_tag.startswith('R'):
        return wn.ADV
    return None
