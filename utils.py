import io
import os
from collections import Counter


def write_to_filepath(text, filepath):
    with io.open(filepath, 'w', encoding='iso-8859-15') as f:
        f.write(text)


def filepath_exists(filepath):
    return os.path.isfile(filepath)


def count_unigrams(string):
    unigrams = Counter()
    for word in string.split():
        increment_dict(word, unigrams, 1)
    return unigrams


def increment_dict(key, dictionary, increment):
    if key not in dictionary:
        dictionary[key] = increment
    else:
        dictionary[key] += increment
