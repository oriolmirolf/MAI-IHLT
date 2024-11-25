# data_loader.py

import os


def load_data(data_dir, dataset_type='train'):
    """
    Load the dataset from the given directory.

    Args:
        data_dir (str): Path to the data directory.
        dataset_type (str): Type of dataset to load ('train' or 'test-gold').

    Returns:
        list of tuples: Each tuple contains (sentence1, sentence2, score, dataset_name)
    """
    input_files = []
    gs_files = []

    if dataset_type == 'train':
        data_path = os.path.join(data_dir, 'train')
    elif dataset_type == 'test':
        data_path = os.path.join(data_dir, 'test-gold')

    input_files = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if file.startswith('STS.input') or file.startswith('STS.input.surprise')
    ]
    gs_files = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if (file.startswith('STS.gs') or file.startswith('STS.gs.surprise')) and not file.endswith('ALL.txt')
    ]

    data = []

    for input_file, gs_file in zip(sorted(input_files), sorted(gs_files)):
        dataset_name = os.path.basename(input_file).split('.')[2]
        if dataset_name.startswith('surprise'):  # Handle special naming convention for "surprise" datasets
            dataset_name = '.'.join(os.path.basename(input_file).split('.')[2:4])
        with open(input_file, 'r', encoding='utf-8') as f_inp, open(
            gs_file, 'r', encoding='utf-8'
        ) as f_gs:
            sentences = [line.strip().split('\t') for line in f_inp]
            scores = [float(line.strip()) for line in f_gs]
            data.extend(
                [
                    (sent_pair[0], sent_pair[1], score, dataset_name)
                    for sent_pair, score in zip(sentences, scores)
                ]
            )

    return data
