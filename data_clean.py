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

import os
import random
import re

import nltk
import numpy as np

from local_loader import sample_file_gen, sample_clean_file
from params import make_csv_dict
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer

import pdb
from stop_words import get_stop_words


class DataCleaner:

    def __init__(self, data_loc, params_loc):
        self.params = make_csv_dict(params_loc)
        self.data_loc = data_loc

        # Put commonly used params into fields
        self.gram_num    = self.params['gram_num']
        self.start_word  = self.params['start_word']
        self.stop_word   = self.params['stop_word']
        self.sample_rate = self.params['sample_rate']
        self.subreddit   = self.params['subreddit']
        self.min_score   = self.params['min_score']

    def load_data(self):
        """
        Purpose: Loads and cleans data specified during init
        Input: Nothing
        Output: Nothing
        """

        # This is for things that require no overarching knowledge
        # of the dataset. For example, removing non-alphanum characters
        print("First stage cleaning...")
        stats = self.clean_data_stage_1(
            sample_file_gen(self.data_loc,
                            self.subreddit,
                            sample_rate=self.sample_rate,
                            flip=False,
                            min_score=self.min_score))

        #model = Word2Vec(corpus_file='./data/clean/RC_2011-01',size=200, window=3, negative=10, min_count=5, workers=4)
        #pdb.set_trace()

        # This is for things that rely on stats collected by the
        # first stage cleaner. For example, removing words of less
        # than a given frequency, which requires having already gone
        # over the data once. 
        print("Second stage cleaning...")
        self.clean_data_stage_2(
            sample_clean_file(stats['loc']),
            stats['unigrams'])
                                        
        
        print("Creating " + str(self.gram_num) + "-grams...")
        self.ngram_dict = self.data_to_grdict(self.raw_str_data,
                                              self.gram_num,
                                              normalize=False)

        print("Creating vocabulary vector")
        self.vocabulary = self.make_voc_vec(self.ngram_dict)

        
    def str_to_ngrams(self, string, gram_num):
        """
        Purpose: Returns a tuple of the n-grams for a string
        Input: string    - String to convert into ngrams
               gram_num  - Number of n-grams to make for the string
        Returns: Tuple of n-grams
        """
        n_grams = nltk.ngrams(nltk.word_tokenize(string), gram_num)
        return [' '.join(n_gram) for n_gram in n_grams]

    def normalize(self, dictionary):
        """
        Purpose: Divides each value in a dictionary by the sum of all values
        Input: Dictionary with number values
        Returns: Dictionary with number values in [0..1]
        """
        minimum = min(dictionary.values())
        total   = max(dictionary.values()) - minimum 
        return {k : (v - minimum) / total \
                for k,v in dictionary.items()}

    def data_to_grdict(self, data, gram_num, normalize=False):
        """
        Purpose: Turns a list of strings into a dictionary of n-gram occurrences
        Input: data      - List of strings
               gram_num  - 'n' in n-gram
               normalize - Divide all dictionary values by the max value?
        
        """
        out_dict = dict()
        
        for comment in data:
            for n_gram in self.str_to_ngrams(comment, gram_num):
                if n_gram in out_dict:
                    out_dict[n_gram] += 1
                else:
                    out_dict[n_gram] = 1

        if normalize:
            out_dict = normalize(out_dict)

        return out_dict

    def data_to_grvec(self, data, vocabulary, gram_num, normalize):
        """
        Purpose: Turns a list of strings into a list of n-gram vectors
        Input: data       - List of strings (bodies of comments)
               vocabulary - List of strings constituting words in the vocabulary
               gram_num   - n in n-grams
               normalize  - Normalize each vector to be a unit vector?
        Output: 
        """
        outvec = np.zeros((len(data), len(vocabulary)))

        for i in range(len(data)):
            outvec[i] = comm_to_grvec(data[i], vocabulary, gram_num, normalize)

        return outvec

    def comm_to_grvec(self, comment, vocabulary, gram_num, normalize):
        """
        Purpose: Takes a single cleaned comment and converts it into a vector of size
                  len(vocabulary).
        Input: comment    - Cleaned comment string
               vocabulary - List of strings constituting words in the vocabulary
               gram_num   - N in n-grams for the vocabulary
               normalize  - If False, returns vector where 1 in position i indicates 
                            presence of a n-gram in position i in the vocabulary
                            If True, normalizes the vector so that it is a unit vector 
        Returns: Vector of size len(vocabulary)
        """
        # Allocate the output vector
        gram_vec = np.zeros(len(vocabulary))
    
        n_grams = [gr for gr in self.str_to_ngrams(comment, gram_num)]
    
        # Loop through all the n-grams
        for gram in n_grams:
            if gram in vocabulary:
                gram_vec[vocabulary.index(gram)] = 1
                # Presence vs not presence probably less noisy than total #
                # gram_vec[vocabulary.index(asonestr)] += 1
            else:
                #print(asonestr)
                pass
    
        return gram_vec
    
    def clean_data_stage_1(self, data):
        """
        Purpose: Given a list of comments, cleans each one
        Input: List of dictionaries corresponding to comments
        Output: List of strings len of which is <= to the len of the input
        """
        loc = os.path.join(os.path.dirname(self.data_loc),
                           '../clean/',
                           os.path.basename(self.data_loc) + "_s1")

        out = open(loc, 'w')

        unigrams = {}
        
        for comment in data:
            comment = self.clean_comment_stage_1(comment)
            if comment != '':
                for word in comment.split():
                    if word not in unigrams:
                        unigrams[word] = 1
                    else:
                        unigrams[word] += 1
                                        
                out.write(comment + '\n')

        out.close()
        
        stats = {}
        stats['unigrams'] = unigrams
        stats['loc'] = loc
        return stats

                
    
    def clean_comment_stage_1(self, comment):
        """
        Purpose: Cleans a given comment, adding stop and start and stop symbols. 
        Input: Comment to be cleaned, as a dictionary
        Returns: Clean comment as a string
                 Returns an empty string if the entire comment was trash
        """
        if self.params['lowercase']:
            comment = comment['body'].lower()
        else:
            comment = comment['body']
            
        comment = self.start_word + " " + comment + " " + self.stop_word

        # Remove deleted comments
        comment = re.sub('((.deleted.)|(.removed.))', ' ', comment).strip()

        # Remove links
        if self.params['remove_links']:
            comment = re.sub('([\[\]])|\(((http:)|(www.)).*\)', ' ', comment).strip()
        
        newcom = ""
        for word in comment.split():
            # Maybe toss links
            if self.params['remove_links'] \
               and ('http:/' in word or 'www.' in word):
                continue
            newcom += word + " "

        # Remove bugged out &<> substitutions
        comment = re.sub('&(amp|gt|lt|nbsp);', ' ', newcom.strip())
        
        if self.params['only_alphanum']:
            comment = re.sub('(\')+', '', comment)
            comment = re.sub('([^ 0-9a-zA-Z])+', '', comment)
    

        comment = ' '.join(comment.split())

        stop_words = get_stop_words('en')
        stop_words = stop_words + ['like', 'just', 'can', 'think', 'will',
                                   'know', 'get', 'really']

        non_stop_words = []
        for word in comment.split():
            if word not in stop_words:
                non_stop_words.append(word)

        comment = ' '.join(non_stop_words)

        # Toss empty comments
        if len(comment) <= len(self.start_word + "  " + self.stop_word):
            return ''

        ps = PorterStemmer()
        stemmed_comment = []
        for w in comment.split():
            stemmed_comment.append(ps.stem(w))

        comment = ' '.join(stemmed_comment)

        return comment

    def clean_data_stage_2(self, data, uni_dict):
        loc = os.path.join(os.path.dirname(self.data_loc),
                           '../clean/',
                           os.path.basename(self.data_loc)[:-3])
        print(loc)

        
        
        pdb.set_trace()


    # @@@@@@@@@@ DEPRECATED @@@@@@@@@@
    def filter_comments(self, data, filt_field, value):
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
               or (self.start_word + ' ' + self.stop_word) in comm['body']:
                continue
    
            comm['body'] = clean_comment(comm)
            newcoms.append(comm)

        return newcoms

    def print_top(self, num):
        """
        Purpose: Prints the top num elements of a dictionary, sorted by descending value
        Input: Number of top values to print
        """
        astoops = [(k, self.ngram_dict[k]) for k in self.ngram_dict]
        astoops = sorted(astoops, key=lambda x: x[1], reverse=True)
    
        for i in range(min(num, len(self.ngram_dict))):
            print("{0:0.5f}".format(astoops[i][1]) \
                  + "\t" + astoops[i][0])
    
    def make_voc_vec(self, dictionary):
        """
        Purpose: Creates a vocabulary vector from a given dictionary
        Input: Dictionary (Usually of unigram frequencies, NOT normalized)
        Output: List of words, sorted
        """
    
        min_freq   = self.params['min_word_count']
        #remove_avg = clean_params['mean_subtract_grams']
    
        voc_vec = list()
        for elem in dictionary:
            if dictionary[elem] > min_freq:
                voc_vec.append(elem)
    
        return sorted(voc_vec)
    
    
    def file_to_grams(data_file_name, params):
        """
    
    
        """
        pass


fleeb = DataCleaner('./data/raw/RC_2007-02', './cfg/clean_params/clean_params.csv')

fleeb.load_data()

pdb.set_trace()

