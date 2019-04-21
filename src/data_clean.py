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

from local_loader import sample_file_gen_multi, sample_clean_file
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
        self.subreddits  = self.params['subreddits']
        self.min_score   = self.params['min_score']

        self.mappings = []

        self.s1_clean_loc = os.path.join(os.path.dirname(self.data_loc),
                                         '../clean/',
                                         os.path.basename(self.data_loc) + "_s1")

        self.s2_clean_loc = os.path.join(os.path.dirname(self.data_loc),
                                         '../clean/',
                                         os.path.basename(self.data_loc) + "_s2")


    def load_data_for_word2vec(self):
        """
        Purpose: Loads and cleans data specified during init
        Input: Nothing
        Output: Nothing
        """

        # This is for things that require no overarching knowledge
        # of the dataset. For example, removing non-alphanum characters
        print("First stage cleaning...")
        stats = self.clean_data_stage_1(
            sample_file_gen_multi(self.data_loc,
                            self.subreddits,
                            min_score=self.min_score))

        # This is for things that rely on stats collected by the
        # first stage cleaner. For example, removing words of less
        # than a given frequency, which requires having already gone
        # over the data once. 
        print("Second stage cleaning...")
        self.clean_data_stage_2(
            sample_clean_file(self.s1_clean_loc),
            stats['unigrams'])

    def create_model(self):
        return Word2Vec(corpus_file=self.s2_clean_loc,size=200, window=3, negative=10, min_count=4, workers=4)

    def make_comment_embeddings(self, model):
        comm_mat = np.ndarray([self.num_s2_comments, model.vector_size], dtype=np.float32)
        one_row = np.zeros([model.vector_size], dtype=np.float32)

        gen = sample_clean_file(self.s2_clean_loc)

        
        for num,comment in enumerate(gen):
            for word in comment:
                if word in model.wv:
                    one_row += model.wv[word]

            if sum(one_row) < .0000001:
                one_row = np.random.random(len(one_row))
                
            one_row /= np.linalg.norm(one_row)
            comm_mat[num] = one_row
            one_row[:] = 0

        return comm_mat

    def make_wv_for_string(self, model, string):
        wv = np.zeros([model.vector_size], dtype=np.float32)
        for word in string.split():
            if word in model.wv:
                wv += model.wv[word]
            
            wv /= np.linalg.norm(wv)

        return wv
        
    def clean_data_stage_1(self, data):
        """
        Purpose: Given a list of comments, cleans each one
        Input: List of dictionaries corresponding to comments
        Output: List of strings len of which is <= to the len of the input
        """

        print("Stage 1 output location", self.s1_clean_loc)


        out = open(self.s1_clean_loc, 'w')

        unigrams = {}

        comm_num = 0
        for comment in data:
            comment = self.clean_comment_stage_1(comment)
            if comment != '':
                for word in comment.split():
                    if word not in unigrams:
                        unigrams[word] = 1
                    else:
                        unigrams[word] += 1

                self.mappings.append(comm_num)

                out.write(comment + '\n')
                
            comm_num += 1
            
        out.close()
        
        stats = {}
        stats['unigrams'] = unigrams
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
        print("Stage 2 output location", self.s2_clean_loc)

        out = open(self.s2_clean_loc, 'w')

        new_maps = []
        num_comments = 0
        
        for comment in data:
            #print(comment)
            comment = self.clean_comment_stage_2(comment, uni_dict)
            if comment != '':
                #for word in comment.split():
                    #if word not in unigrams:
                    #    unigrams[word] = 1
                    #else:
                    #    unigrams[word] += 1

                new_maps.append(self.mappings[num_comments])
            
                out.write(comment + '\n')
            num_comments += 1

            
        self.mappings = new_maps
        
        out.close()
        self.num_s2_comments = len(new_maps)
        
    def clean_comment_stage_2(self, comment, uni_dict):
        """
        Purpose: Cleans a single comment, mainly removing words that occur
                 less than a given number of times
        Input: A single comment and the unigram frequency dictionary
        Output: A cleaned comment
        """
        newCom = [word for word in comment \
                  if (word in uni_dict
                      and uni_dict[word] >= self.params['min_word_count'])]

        return " ".join(newCom)

