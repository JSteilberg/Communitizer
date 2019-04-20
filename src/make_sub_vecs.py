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
import datetime
import pdb

import utils
from cluster import Clusternator

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


def main():
    start_time = datetime.datetime.now().time()
    cnator = Clusternator(FILE, CLEAN, 10)
    cnator.prepare_data()

    model = cnator.model

    sub_embed_dict = cnator.dc.create_sub_embed_dict(SUBS, cnator.model, ALL_SAMP_RATE, NUM_WORDS)

    print("Starting Clustering", str(datetime.datetime.now().time()))
    cnator.run_k_means()
    print("Finished Clustering", str(datetime.datetime.now().time()))
    cluster_commonword_dict = cnator.get_clusterwords(15)
    utils.write_to_filepath(str(cluster_commonword_dict), "clusterwords.txt")
    cnator.dc.df.to_csv('./clusters.csv')

    print("Calculating cluster subreddit similarity...")
    sim_df, cluster_subreddit_labels = cnator.get_subreddit_similarity(sub_embed_dict, model, 10)
    score = cnator.evaluate_cluster(cluster_subreddit_labels)
    print("Overall start time:", start_time)
    print("Overall end time: ", datetime.datetime.now().time())
    print(str(cluster_subreddit_labels))
    print(str(score))
    pdb.set_trace()


main()
