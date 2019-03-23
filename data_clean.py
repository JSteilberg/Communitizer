################################################################################
# Purpose: Cleans, filters, and creates ngrams for the data
# Authors: Jack Steilberg <jsteilberg@ccis.neu.edu>,
#          Sajid Raihan   <raihan.s@husky.neu.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
################################################################################

import random
import re

import nltk
import numpy as np

from local_loader import sample_file

import pdb

def str_to_ngrams(string, gram_num):
    """
    Purpose: Returns a tuple of the n-grams for a string
    Input: string    - String to convert into ngrams
           gram_num  - Number of n-grams to make for the string
    Returns: Tuple of n-grams
    """
    n_grams = nltk.ngrams(nltk.word_tokenize(string), gram_num)
    return  [' '.join(n_gram) for n_gram in n_grams]

def normalize(dictionary):
    """
    Purpose: Divides each value in a dictionary by the sum of all values
    Input: Dictionary with number values
    Returns: Dictionary with number values in [0..1]
    """
    minimum = min(dictionary.values())
    total   = max(dictionary.values()) - minimum 
    return {k : (v - minimum) / total \
            for k,v in dictionary.items()}

def data_to_grdict(data, gram_num, normalize=False):
    """
    Purpose: Turns a list of strings into a dictionary of n-gram occurrences
    Input: data      - List of strings
           gram_num  - n in n-grams (e.g. 1 => unigrams, 2 => bigrams)
           normalize - Divide all dictionary values by the max value?
    """
    out_dict = dict()
    
    for comment in data:
        for n_gram in str_to_ngrams(comment, gram_num):
            if n_gram not in out_dict:
                out_dict[n_gram] = 1
            else:
                out_dict[n_gram] += 1

    if normalize:
        out_dict = normalize(out_dict)

    return out_dict

def data_to_grvec(data, gram_num, vocabulary, normalize):
    """
    Purpose: Turns a list of strings into a list of n-gram vectors
    Input: data      - List of strings (bodies of comments)
           gram_num  - n in n-grams
           normalize - Normalize each vector to be a unit vector?
    Output: 
    """
    outvec = np.zeros((len(data), len(vocabulary)))

    for i in range(len(data)):
        outvec[i] = comm_to_grvec(data[i], gram_num, vocabulary, normalize)

    return outvec

def make_vocab(subreddit_dict, overall_dict):
    pass

def comm_to_grvec(comment, gram_num, vocabulary, normalize):
    """
    Purpose: Takes a single cleaned comment and converts it into a vector of size
             len(vocabulary).
    Input: comment    - Cleaned comment string
           gram_num   - Number of ngrams to use when generating vector
           vocabulary - Vocabulary of n-grams 
           normalize  - If False, returns vector where 1 in position i indicates 
                        presence of a n-gram in position i in the vocabulary
                        If True, normalizes the vector so that it is a unit vector 
    Returns: Vector of size len(vocabulary)
    """
    # Allocate the output vector
    gram_vec = np.zeros(len(vocabulary))

    n_grams = [gr for gr in str_to_ngrams(comment, gram_num)]

    # Loop through all the n-grams
    for gram in n_grams:
        if gram in vocabulary:
            gram_vec[vocabulary.index(gram)] = 1
            # Presence vs not presence probably more important than total #
            # gram_vec[vocabulary.index(asonestr)] += 1
        else:
            #print(asonestr)
            pass

    return gram_vec

def clean_data(data):
    """
    Purpose: Given a list of comments, cleans each one
    Input: List of dictionaries corresponding to comments
    Output: List of strings len of which is <= to the len of the input
    """
    out_data = list()
    for comment in data:
        comment = clean_comment(comment)
        if comment != '':
            out_data.append(comment)

    return out_data

def clean_comment(comment):
    """
    Purpose: Cleans a given comment, adding stop and start and stop symbols. 
    Input: Comment to be cleaned, as a dictionary
    Returns: Clean comment as a string
             Returns an empty string if the entire comment was trash
    """
    comment = "|st1 " + comment['body'].lower() + " sp1#"
    comment = comment.replace("deleted", " ")
    comment = comment.replace("removed", " ")

    comment = re.sub('[^0-9a-zA-Z/\:\'\.\+\- ]+', ' ', comment)

    #comment = comment.replace("gt", " ")
    #comment = comment.replace("amp", " ")
    
    comment = ' '.join(comment.split())
    if len(comment) < 8:
        return ''
    else:
        return comment

def filter_comments(data, filt_field, value):
    """
    Purpose: Filters comments, returning those who match the input requirements.
    Input: data       - List of comments (dictionaries)
           filt_field - Field in the comment dictionary to filter on 
           value      - Value for the field
    Returns: List of comments that meet given criteria
    """
    if type(data) == str:
        comms = [k for k in data if k[filt_field].lower() == value]
    else:
        comms = [k for k in data if k[filt_field] == value]

    newcoms = list()
    for comm in comms:
        if '[deleted]' in comm['body'] \
           or '[removed]' in comm['body'] \
           or 'st1 spl' in comm['body']:
            continue

        comm['body'] = clean_comment(comm)
        newcoms.append(comm)

    return newcoms

def print_top(dictionary, num):
    """
    Purpose: Prints the top num elements of a dictionary, sorted by descending value
    Input: dictionary - Dictionary to print
           num        - Number of top values to print
    """
    astoops = [(k, dictionary[k]) for k in dictionary]
    astoops = sorted(astoops, key=lambda x: x[1], reverse=True)

    for i in range(min(num, len(dictionary))):
        print("{0:0.5f}".format(astoops[i][1]) \
              + "\t" + astoops[i][0])

def main():
    data_file = "./test.dat"
    subreddit = "programming"
    gram_num = 1
    
    print("Loading data...")
    data = sample_file(data_file, subreddit, sample_rate=1, flip=False, min_score=0)
    data = clean_data(data)
    gr_dict = data_to_grdict(data, 1, normalize=False)
    gr_vec = data_to_grvec(data, 1, gr_dict.keys(), False)

    print("Creating ngram dictionaries")
    #gen_dict = 
    
    pdb.set_trace()
    gen_norm = normalize(general_dict)
    general_dict.clear()
    print("Size of general dictionary: " + str(len(gen_norm)))
    
    print("General reddit: ")
    print_top(gen_norm, 35)

    #print("\n/r/" + subreddit + ": ")
    #print_top(common, 35)

    #prog_comms = filter_comments(data, 'subreddit', 'politics')
    #funn_comms = filter_comments(data, 'subreddit', 'programming')

    #vocab = sorted([k for k in gen_norm.keys()])

    #print("Making comment vectors")
    #prog_vecs = [make_single_comm_grvec(comm, 1, vocab) for comm in prog_comms[:50]]
    #funn_vecs = [make_single_comm_grvec(comm, 1, vocab) for comm in funn_comms[:50]]

    prog_vecs =([vec / sum(vec) for vec in prog_vecs])
    funn_vecs =([vec / sum(vec) for vec in funn_vecs])

    tot = prog_vecs + funn_vecs

    print("Fitting spherical kmeans")
    skm = SphericalKMeans(n_clusters=2)
    skm.fit(tot)

    for idx,comm in enumerate(prog_comms[:50] + funn_comms[:50]):
        print(skm.labels_[idx], ":",comm['body'][:100])

        

    pdb.set_trace()
        
main()

