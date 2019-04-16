import io
import os
from collections import Counter
import numpy as np


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


def get_top_n_words(df, n):
    """
    Purpose: Given some df, it determines the top n top occurring words
    :param df:
    :param n:
    :return: [] n-length with the top occurring words
    """
    all_counts = Counter()
    for row in df.itertuples():
        cleaned_comment = getattr(row, "Cleaned_Comment")
        all_counts += count_unigrams(cleaned_comment)
    most_common_words = all_counts.most_common(n)

    return [most_common_words[i][0] for i in range(0, len(most_common_words))]


def get_embedding(model, words):
    vect = np.zeros(model.vector_size, dtype=np.float32)
    for word in words:
        if word in model.wv:
            vect += model[word]

    vect /= np.linalg.norm(vect)
    return vect


def make_df_embedding(df, model, n):
    """
    :param df: DF to make embedding of
    :param model: Word2Vec Model to use
    :param n: Number of words ot consider per cluster
    :return: Word2Vec Vector representing the DF
    """
    top_words = get_top_n_words(df, n)
    return get_embedding(model, top_words)


def get_top_keys(uni_dict, num):
    topnum =  sorted(uni_dict.items(),
                     key=lambda x: x[1],
                     reverse=True)[:num]
    return [k[0] for k in topnum]


def convert_lol_to_numpy(lol):
    rol = []
    for l in lol:
        rol.append(np.array(l))
    return np.array(rol)
