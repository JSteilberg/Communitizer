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
import re
import pandas as pd
import numpy as np
from local_loader import sample_file_gen_multi
from params import make_csv_dict
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from stop_words import get_stop_words
import gc
import utils


class DataCleanerDF:

    def __init__(self, data_loc, params_loc):
        self.params = make_csv_dict(params_loc)
        self.data_loc = data_loc

        # Put commonly used params into fields
        self.gram_num = self.params['gram_num']
        self.start_word = self.params['start_word']
        self.stop_word = self.params['stop_word']
        self.sample_rate = self.params['sample_rate']
        self.subreddits = self.params['subreddits']
        self.min_score = self.params['min_score']

        self.df = None

        self.s2_df_clean_loc = os.path.join(os.path.dirname(self.data_loc),
                                            '../clean/',
                                            os.path.basename(self.data_loc) + "_df_s2")

        self.original_comments = None
        self.clean_comments_loc = os.path.join(os.path.dirname(self.data_loc),
                                               '../clean/',
                                               os.path.basename(self.data_loc) + "_clean_comments")
        self.embedded_comments = None

    def load_data_for_word2vec(self):
        """
        Purpose: Loads and cleans data specified during init
        Input: Nothing
        Output: Nothing
        """

        # This is for things that require no overarching knowledge
        # of the dataset. For example, removing non-alphanum characters
        print("First stage cleaning...")
        stats, cleaned_comments_s1 = self.clean_data_stage_1(sample_file_gen_multi(self.data_loc,
                                                                                   self.subreddits,
                                                                                   min_score=self.min_score))

        # This is for things that rely on stats collected by the
        # first stage cleaner. For example, removing words of less
        # than a given frequency, which requires having already gone
        # over the data once.
        print("Second stage cleaning...")
        self.clean_data_stage_2(stats['unigrams'], cleaned_comments_s1)

    def create_model(self):
        if self.df is None:
            raise RuntimeError("Trying to create model without cleaning comments first")
        corpus_file = "\n".join(self.df['Cleaned_Comment'])
        utils.write_to_filepath(corpus_file, self.clean_comments_loc)
        return Word2Vec(corpus_file=self.clean_comments_loc, size=200, window=3, negative=10, min_count=0, workers=4)

    def make_comment_embeddings(self, model):
        num_comments = len(self.df['Cleaned_Comment'])
        comm_mat = np.ndarray([num_comments, model.vector_size], dtype=np.float32)

        row_num = 0
        embeddings_array = []
        e = 1
        len_df = len(self.df.index)
        for idx, comment in enumerate(self.df['Cleaned_Comment']):
            one_row = np.zeros([model.vector_size], dtype=np.float32)
            has_model_words = False

            for word in comment.split():
                if word in model.wv:
                    has_model_words = True
                    one_row += model.wv[word]

            one_row /= np.linalg.norm(one_row)
            comm_mat[row_num] = one_row
            row_num += 1

            if not has_model_words:
                print("very bad")
            embeddings_array.append(one_row)
            if idx == e:
                print(str(idx), "Comments Embedded", "Out of", str(len_df))
                e = e * 2
        print("All", str(len_df), "comments embedded")

        self.embedded_comments = embeddings_array

    def clean_data_stage_1(self, data):
        """
        Purpose: Given a list of comments, cleans each one
        Input: List of dictionaries corresponding to comments
        Output: List of strings len of which is <= to the len of the input
        """

        print("Stage 1")

        unigrams = {}

        subreddit_array = []
        original_comment_array = []
        cleaned_comments_s1 = []

        i = 0
        e = 1

        for json in data:
            subreddit = json['subreddit']
            og_comment = json['body']
            comment = self.clean_comment_stage_1(json)
            if comment != '':
                for word in comment.split():
                    if word not in unigrams:
                        unigrams[word] = 1
                    else:
                        unigrams[word] += 1
                subreddit_array.append(subreddit)
                original_comment_array.append(og_comment)
                cleaned_comments_s1.append(comment)
            if i == e:
                print(str(i), "Comments Cleaned S1")
                e = e * 2
            i += 1
        print("All", str(i), "Comments Cleaned S1")

        data = {'Subreddit': subreddit_array,
                'Original': original_comment_array
                }

        df = pd.DataFrame(data, columns=('Subreddit', 'Original'))
        df.Subreddit = df.Subreddit.astype('category')
        print("S1 Dataframe Created")
        self.df = df
        stats = dict()
        stats['unigrams'] = unigrams
        return stats, cleaned_comments_s1

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

    def clean_data_stage_2(self, uni_dict, cleaned_comments_s1):
        print("Stage 2")

        num_comments = 0

        subreddit_array = []
        original_comment_array = []
        cleaned_comments_s2 = []
        e = 1
        len_df = len(self.df.index)
        for idx, row in enumerate(self.df.itertuples()):
            comment_s1 = cleaned_comments_s1[idx].split()
            subreddit = getattr(row, "Subreddit")
            original_comment = getattr(row, "Original")
            comment_s2 = self.clean_comment_stage_2(comment_s1, uni_dict)

            if comment_s2 != '':
                subreddit_array.append(subreddit)
                original_comment_array.append(original_comment)
                cleaned_comments_s2.append(comment_s2)
            num_comments += 1
            if idx == e:
                print(str(idx), "Comments Cleaned S2", "Out of", str(len_df))
                e = e * 2
        print("All", str(len_df), "Comments Cleaned S2")

        data = {
                'Subreddit': subreddit_array,
                'Cleaned_Comment': cleaned_comments_s2
                }

        df = pd.DataFrame(data, columns=['Subreddit', 'Cleaned_Comment'])
        print("S2 Dataframe Created")
        self.df = df
        self.original_comments = original_comment_array
        gc.collect()  # ensure previous df is gone from memory
        # self.cleaned_comments_s2 = cleaned_comments_s2

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

    def create_sub_embed_dict(self, model, all_samp_rate, num_words):
        all_subs = self.df.Subreddit.unique().tolist()
        subs = []
        for s in all_subs:
            sub_count = len(self.df[(self.df['Subreddit'] == s)])
            if sub_count > 100:
                subs.append(s)
        subs.append('all')
        sub_dict = dict(zip(subs, np.append(np.ones(len(subs) - 1), all_samp_rate)))

        uni_dict = dict()
        for key in subs:
            uni_dict[key] = dict()

        print("Calculating unique unigrams per subreddit and normalizing...")

        i = 0
        e = 1
        # Loop over all the comments
        for comment in sample_file_gen_multi(self.data_loc, subreddits=sub_dict, min_score=2):
            subreddit = comment['subreddit']
            clean = self.clean_comment_stage_1(comment)

            # Go through each word in the cleaned comment, adding it to the
            # unigram dictionary for both its relevant subreddit (if applicable)
            # and the all-of-reddit unigram dictionary
            for word in clean.split():
                # Relevant subreddit
                if subreddit in uni_dict:
                    if word in uni_dict[subreddit]:
                        uni_dict[subreddit][word] += 1.0
                    else:
                        uni_dict[subreddit][word] = 1.0

                # All of reddit
                if word in uni_dict['all']:
                    uni_dict['all'][word] += 1.0
                else:
                    uni_dict['all'][word] = 1.0
            if i == e:
                print(str(i), "Comments Counted")
                e = e * 2
            i += 1
        print("All", str(i), "Comments Counted")

        print("Normalizing subreddit unigram counts")

        for subreddit in uni_dict:
            max = -float('inf')
            for word in uni_dict[subreddit]:
                if uni_dict[subreddit][word] > max:
                    max = uni_dict[subreddit][word]

            for word in uni_dict[subreddit]:
                uni_dict[subreddit][word] /= max

        print("Calculating set difference of all unigrams from subreddits ")

        for subreddit in uni_dict:
            if subreddit == 'all':
                continue

            for word in uni_dict[subreddit]:
                uni_dict[subreddit][word] -= 1.0 * uni_dict['all'][word]

        embed_dict = dict()

        print("Creating embeddings per subreddit...")
        for subreddit in uni_dict:
            vect = np.zeros(model.vector_size, dtype=np.float32)
            for word in utils.get_top_keys(uni_dict[subreddit], num_words):
                if word in model.wv:
                    vect += model[word]

            # This will sometimes happen if you're processing a dataset using a
            # Word2Vec model trained on a different dataset. Ideally, it won't
            # happen too much, and if it does we just turn the vector into a random
            # noise vector to avoid biasing the results.
            if np.linalg.norm(vect) < .00000001:
                vect = np.random.random(len(vect))
                print("Bad! This shouldn't happen more than 20 times.")

            vect /= np.linalg.norm(vect)
            embed_dict[subreddit] = vect

        return embed_dict
