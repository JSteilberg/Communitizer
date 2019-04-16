################################################################################
# Purpose: Creates an embedding vector to represent given subreddits
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

import numpy as np
import pdb
from gensim.models import Word2Vec
from pandas import Series
from local_loader import sample_file_gen_multi
from data_clean_df import DataCleanerDF
from cluster import Clusternator
import utils

FILE = 'RC_2015-06_sub'
CLEAN = './cfg/' + 'clean_params/clean_params.csv'
NUM_WORDS = 30
# SUBS = ['politics', 'programming', 'science']
SUBS = ['programming', 'politics', 'niggers', 'groids', 'fatpeoplehate']

ALL_SAMP_RATE = .2


def print_top(uni_dict, num):
    for subreddit in uni_dict:
        print(subreddit + ":")
        topnum =  sorted(uni_dict[subreddit].items(),
                         key=lambda x: x[1],
                         reverse=True)[:num]
        for word in topnum:
            print("    " + str(word[0]) + ": " + str(word[1]))


def create_sub_embed_dict(dc, file, model):
    SUBS.append('all')
    sub_dict = dict(zip(SUBS, np.append(np.ones(len(SUBS) - 1), ALL_SAMP_RATE)))

    uni_dict = dict()
    for key in SUBS:
        uni_dict[key] = dict()

    print("Calculating unique unigrams per subreddit and normalizing...")

    # Loop over all the comments
    for comment in sample_file_gen_multi(file, subreddits=sub_dict, min_score=2):
        subreddit = comment['subreddit']
        clean = dc.clean_comment_stage_1(comment)

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

    for subreddit in uni_dict:

        max = -float('inf')
        for word in uni_dict[subreddit]:
            if uni_dict[subreddit][word] > max:
                max = uni_dict[subreddit][word]

        for word in uni_dict[subreddit]:
            uni_dict[subreddit][word] /= max

    for subreddit in uni_dict:
        if subreddit == 'all':
            continue

        for word in uni_dict[subreddit]:
            uni_dict[subreddit][word] -= 1.0 * uni_dict['all'][word]

    embed_dict = dict()

    print("Creating embeddings per subreddit...")
    for subreddit in uni_dict:
        vect = np.zeros(model.vector_size, dtype=np.float32)
        for word in utils.get_top_keys(uni_dict[subreddit], NUM_WORDS):
            if word in model.wv:
                vect += model[word]

        vect /= np.linalg.norm(vect)
        embed_dict[subreddit] = vect

    return embed_dict


def main():
    cnator = Clusternator(FILE, CLEAN, 10)
    cnator.prepare_data()

    print("Loading Word2Vec model...")
    model = cnator.model

    sub_embed_dict = cnator.dc.create_sub_embed_dict(SUBS, cnator.model, ALL_SAMP_RATE, NUM_WORDS)

    cnator.run_k_means()

    print("Calculating cluster subreddit similarity...")
    sim_df, cluster_subreddit_labels = cnator.get_subreddit_similarity(sub_embed_dict, model, 10)
    pdb.set_trace()
    score = cnator.evaluate_cluster(cluster_subreddit_labels)


main()
