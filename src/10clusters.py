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
CLEAN = '../cfg/' + 'clean_params/clean_params_6_clusters.csv'
NUM_WORDS = 30

ALL_SAMP_RATE = .15


def main():
    start_time = datetime.datetime.now().time()
    cnator = Clusternator(FILE, CLEAN, 10)
    cnator.prepare_data()

    model = cnator.model

    sub_embed_dict = cnator.dc.create_sub_embed_dict(model, ALL_SAMP_RATE, NUM_WORDS)

    print("Starting Clustering", str(datetime.datetime.now().time()))
    cnator.spherical_k_means()
    print("Finished Clustering", str(datetime.datetime.now().time()))
    cluster_commonword_dict = cnator.get_clusterwords(6)
    utils.write_to_filepath(str(cluster_commonword_dict), "10clusterwords.txt")
    print("Creating clusters.csv", datetime.datetime.now().time())
    cnator.dc.training_df.to_csv('./10clusters.csv')
    print("Created clusters.csv", datetime.datetime.now().time())

    print("Calculating cluster subreddit similarity...")
    sim_df, cluster_subreddit_labels = cnator.get_subreddit_similarity(sub_embed_dict, model, 10)
    total_acc = cnator.evaluate_cluster(cluster_subreddit_labels)

    cnator.get_cluster_stats()
    print("Overall start time:", start_time)
    print("Overall end time: ", datetime.datetime.now().time())
    print(str(cluster_subreddit_labels))
    print("total accuracy", total_acc)
    pdb.set_trace()


main()
