# preprocessing.py

import nltk
import string
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))


def preprocess_sentence(sentence):
    """
    Preprocess the input sentence.

    Args:
        sentence (str): The sentence to preprocess.

    Returns:
        list: A list of lemmatized tokens.
    """
    # Lowercase
    sentence = sentence.lower()
    # Tokenize
    tokens = nltk.word_tokenize(sentence)
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # Remove stopwords is worse!
    # tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    return tokens


def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts.

    Args:
        word (str): The word to get POS tag for.

    Returns:
        str: WordNet POS tag
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}

    return tag_dict.get(tag, wn.NOUN)


def get_pos_tags(tokens):
    """
    Get POS tags for the tokens.

    Args:
        tokens (list): List of tokens.

    Returns:
        list: List of POS tags.
    """
    pos_tags = nltk.pos_tag(tokens)
    # Return only the tags
    return [tag for word, tag in pos_tags]


def get_dependency_relations(sentence):
    """
    Get dependency relations from the sentence using spaCy.

    Args:
        sentence (str): The input sentence.

    Returns:
        list: List of dependency relations (dep, head text, token text)
    """
    doc = nlp(sentence)
    dependencies = [(tok.dep_, tok.head.text, tok.text) for tok in doc]
    return dependencies
