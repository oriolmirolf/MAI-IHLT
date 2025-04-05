import nltk
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from gensim import corpora, matutils
from deep_translator import GoogleTranslator
from nltk.corpus import brown

from collections import Counter, defaultdict

from nltk.corpus import brown, wordnet as wn, wordnet_ic
from nltk import pos_tag

import scipy

import numpy as np

from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning,
    module="sklearn.decomposition._truncated_svd"
    )

# ---------------------------- #
# ESA MODEL IMPLEMENTATION     #
# ---------------------------- #
#
# https://github.com/GermanT5/wikipedia2corpus?tab=readme-ov-file#wikipedia-2-corpus #



class ESA_Model:
    def __init__(self):
        """Initialize the ESA model by loading or building the term-concept matrix."""
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

        if self._model_exists():
            print("Pre-trained ESA model found. Loading...")
            self.esa_index = self._load_model()
        else:
            print("No pre-trained ESA model found. Building the model...")
            self.esa_index = self.build_esa_index()

    def _model_exists(self):
        """Check if the ESA model files exist."""
        return os.path.exists("models/esa/esa_index.npz") and os.path.exists("models/esa/esa_terms.pkl")

    def _load_model(self):
        """Load the pre-trained ESA model."""
        esa_index = scipy.sparse.load_npz("models/esa/esa_index.npz")
        with open("models/esa/esa_terms.pkl", "rb") as f:
            esa_terms = pickle.load(f)
        return {"index": esa_index, "terms": esa_terms}

    def build_esa_index(self):
        """
        Build the ESA index from Wikipedia corpus sequentially.

        Returns:
            dict: A dictionary mapping terms to concept vectors.
        """
        print("Building ESA index from Wikipedia...")

        corpus_file = "enwiki-20220201-clean.txt"
        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"Corpus file '{corpus_file}' not found.")

        class MyCorpus:
            def __init__(self, max_documents=100_000):
                self.max_documents = max_documents

            def __iter__(self):
                document = []
                document_count = 0
                with open(corpus_file, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped_line = line.strip()
                        if not stripped_line and document: # end of article
                            yield document
                            document = []
                            document_count += 1
                            if document_count >= self.max_documents:
                                break
                        elif stripped_line:
                            document.append(stripped_line)
                if document and document_count < self.max_documents:
                    yield document

        corpus = MyCorpus()
        dictionary_path = "dictionary.pkl"
        bow_path = "bow_corpus.mm"

        # step 1: Build and finalize the dictionary
        if os.path.exists(dictionary_path):
            with open(dictionary_path, "rb") as f:
                dictionary = pickle.load(f)
        else:
            print("Building dictionary...")
            dictionary = corpora.Dictionary()
            for doc in tqdm(corpus, desc="Processing documents for dictionary"):
                processed_doc = self._process_document(doc)
                dictionary.add_documents([processed_doc])
            dictionary.filter_extremes(no_below=5, no_above=0.5)
            dictionary.compactify()
            with open(dictionary_path, "wb") as f:
                pickle.dump(dictionary, f)

        # step 2: build the sparse term-document matrix
        print("Building sparse term-document matrix...")
        if os.path.exists(bow_path):
            bow_corpus = corpora.MmCorpus(bow_path)
        else:
            bow_corpus = []
            for doc in tqdm(corpus, desc="Processing documents for corpus"):
                processed_doc = self._process_document(doc)
                bow = dictionary.doc2bow(processed_doc)
                bow_corpus.append(bow)

            corpora.MmCorpus.serialize(bow_path, bow_corpus)

        # step 3: convert to term-document matrix
        print("Converting corpus to term-document matrix...")
        td_matrix = matutils.corpus2csc(bow_corpus, num_terms=len(dictionary))

        # step 4: apply Truncated SVD
        print("Applying Truncated SVD...")
        svd = TruncatedSVD(n_components=500, random_state=42)
        term_concept_matrix = svd.fit_transform(td_matrix)

        print("Saving ESA index...")
        scipy.sparse.save_npz("models/esa/esa_index.npz", scipy.sparse.csr_matrix(term_concept_matrix))
        with open("models/esa/esa_terms.pkl", "wb") as f:
            pickle.dump(dictionary.token2id, f)

        return {"index": scipy.sparse.csr_matrix(term_concept_matrix), "terms": dictionary.token2id}

    def _process_document(self, doc):
        """
        Process a document by tokenizing, lemmatizing, and removing stopwords.

        Args:
            doc (list of str): Sentences in a document.

        Returns:
            list of str: Tokenized, lemmatized, and cleaned tokens.
        """
        tokens = []
        for sentence in doc:
            words = nltk.word_tokenize(sentence)
            for word in words:
                word = word.lower()
                if word.isalpha() and word not in self.stopwords:
                    lemma = self.lemmatizer.lemmatize(word)
                    tokens.append(lemma)
        return tokens

    def compute_similarity(self, text1, text2):
        """
        Compute the ESA similarity between two texts.

        Args:
            text1 (str): First text.
            text2 (str): Second text.

        Returns:
            float: ESA similarity score.
        """

        def get_vector(text):
            # preprocess as ukp here
            tokens = [
                self.lemmatizer.lemmatize(w.lower())
                for w in nltk.word_tokenize(text)
                if w.isalpha() and w.lower() not in self.stopwords
            ]

            term_ids = [self.esa_index["terms"][t] for t in tokens if t in self.esa_index["terms"]]

            if not term_ids:
                return np.zeros(self.esa_index["index"].shape[1])

            # sum the corresponding rows in the ESA index (each row corresponds to a term)
            # .sum(axis=0) gives a 1xN sparse matrix; .A1 converts it to a 1D numpy array
            concept_vector = self.esa_index["index"][term_ids, :].sum(axis=0).A1

            # normalize the concept vector
            norm = np.linalg.norm(concept_vector)
            return concept_vector / norm if norm > 0 else concept_vector

        vec1 = get_vector(text1)
        vec2 = get_vector(text2)
        sim = np.dot(vec1, vec2)
        return sim
    
    
    
# --------------------------- #
# LEXICAL SUBSTITUTION MODEL  #
# --------------------------- #

class LexSubModel:
    def __init__(self):
        """Initialize the lexical substitution model."""
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.frequent_nouns = self.get_frequent_nouns()

    def get_frequent_nouns(self):
        """Retrieve a set of about 1,000 frequent English nouns."""
        words = [w.lower() for w in brown.words() if w.isalpha()]
        tagged = nltk.pos_tag(words)
        nouns = [w for w, pos in tagged if pos.startswith('NN')]
        freq_dist = Counter(nouns)
        most_common_nouns = [word for word, freq in freq_dist.most_common(1000)]
        return set(most_common_nouns)

    def get_substituted_tokens(self, s_proc):
        """Get substituted tokens for the sentence."""
        substituted_tokens = []
        pos_tagged_tokens = s_proc['pos_tagged_tokens']
        for token, pos in pos_tagged_tokens:
            if pos.startswith('NN'):
                lemma = self.lemmatizer.lemmatize(token.lower())
                if lemma in self.frequent_nouns:
                    synonyms = self.get_synonyms(lemma)
                    if synonyms:
                        substituted_tokens.extend(synonyms)
        return substituted_tokens

    def get_synonyms(self, word):
        """Get synonyms from WordNet for the given word."""
        synonyms = set()
        for syn in wn.synsets(word, pos=wn.NOUN):
            for lemma in syn.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                if name != word:
                    synonyms.add(name)
        return list(synonyms)
    
    
    
# ---------------------------- #
# SMT BACK-TRANSLATION MODEL   #
# ---------------------------- #


class SMT_Model:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.bridge_languages = ['de', 'es', 'nl']
    
    def backtranslate(self, text):
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [
            self.lemmatizer.lemmatize(token.lower()) 
            for token in tokens 
            if token.lower() not in self.stopwords
        ]
        filtered_text = ' '.join(filtered_tokens)

        back_translated_texts = []
        for lang in self.bridge_languages:
            
            # bridge lang
            foreign_text = GoogleTranslator(source='en', target=lang).translate(filtered_text)

            # back to english
            back_trans_text = GoogleTranslator(source=lang, target='en').translate(foreign_text)
            back_translated_texts.append(back_trans_text)

        return ' '.join(back_translated_texts)
    
    

# -----------------------------#
# DISTRIBUTIONAL THESAURUS     #
# -----------------------------#

class DistributionalThesaurus:
    def __init__(self):
        """Initialize the Distributional Thesaurus."""

        self.cooccurrence_matrix = self.build_cooccurrence_matrix()

    def build_cooccurrence_matrix(self):
        """Build a co-occurrence matrix from the Brown corpus focusing on cardinal numbers."""

        cooccurrence = defaultdict(lambda: defaultdict(int))
        window_size = 2
        for sent in brown.sents():
            tagged_sent = pos_tag(sent)
            tokens_cd = [token.lower() for token, pos in tagged_sent if pos == 'CD']
            for i, word in enumerate(tokens_cd):
                context_indices = range(max(i - window_size, 0), min(i + window_size + 1, len(tokens_cd)))
                for j in context_indices:
                    if i != j:
                        cooccurrence[word][tokens_cd[j]] += 1
        return cooccurrence

    def dt_similarity(self, tokens1, tokens2):
        """Compute similarity based on Distributional Thesaurus focusing on cardinal numbers."""
        # only (POS tag 'CD')
        tokens1_cd = [token.lower() for token, pos in pos_tag(tokens1) if pos == 'CD']
        tokens2_cd = [token.lower() for token, pos in pos_tag(tokens2) if pos == 'CD']
        if not tokens1_cd or not tokens2_cd:
            return 0.0
        sims = []
        for w1 in tokens1_cd:
            for w2 in tokens2_cd:
                sim = self.compute_similarity(w1, w2)
                sims.append(sim)
        if sims:
            return np.mean(sims)
        else:
            return 0.0

    def compute_similarity(self, w1, w2):
        """Compute co-occurrence similarity between two words."""
        vec1 = self.cooccurrence_matrix.get(w1, {})
        vec2 = self.cooccurrence_matrix.get(w2, {})
        if not vec1 or not vec2:
            return 0.0

        common_words = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[word] * vec2[word] for word in common_words)
        sum1 = sum(count ** 2 for count in vec1.values())
        sum2 = sum(count ** 2 for count in vec2.values())
        denominator = np.sqrt(sum1) * np.sqrt(sum2)
        if denominator == 0:
            return 0.0
        return numerator / denominator