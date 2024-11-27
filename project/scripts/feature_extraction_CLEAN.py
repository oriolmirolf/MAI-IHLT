# feature_extraction.py

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from feature_extraction.feature_utils import extract_features


class FeatureExtractor:
    def __init__(self):
        """Initialize the FeatureExtractor with an internal memoization dictionary."""
        self.memoized_pair_features = {}

    def extract_pair_features(self, s1, s2):
        """Extract features for a pair of sentences with order-invariant memoization."""
        key = frozenset([s1, s2])
        if key in self.memoized_pair_features:
            return self.memoized_pair_features[key]
        features = extract_features(s1, s2)
        self.memoized_pair_features[key] = features
        return features

    def process_row(self, row):
        """Process a single row and extract features for the sentence pair."""
        s1, s2, score, ds = row

        # Extract features for the sentence pair
        features = self.extract_pair_features(s1, s2)

        # Include score and dataset in the result
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
