################################################################################
# Purpose: Runs clustering algorithm over cleaned dataset
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
from collections import Counter
from spherecluster import SphericalKMeans
import utils
import pdb


class Clusternator:

    def __init__(self, data, num_clusters):
        self.data = data
        self.n_cluster = num_clusters
        self.skm = SphericalKMeans(n_clusters=self.n_cluster)

    def run_k_means(self):
        self.skm.fit(self.data)

        return self.skm

    def get_clusterwords(self, df, n_most_common):
        cluster_commonword_dict = dict()
        for c_num in range(0, self.n_cluster):
            cluster_df = df.loc[df['Cluster_Num'] == c_num]
            cluster_commonwords = Counter()
            for row in cluster_df.itertuples():
                comment = getattr(row, "Cleaned_Comment")
                cluster_commonwords += utils.count_unigrams(comment)
            most_common_words_counts = cluster_commonwords.most_common(n_most_common)
            most_common_words = []
            for i in range(0, len(most_common_words_counts)):
                most_common_words.append(most_common_words_counts[i][0])
            cluster_commonword_dict[c_num] = most_common_words
        return cluster_commonword_dict

    def get_cluster_subreddit(self, df):
        """
        Purpose: Given some df, it determines the subreddit
                 of each of the clusters
        Input: Given a df, determine the subreddit of each cluster
               (based off of majority)
        Output: Dictionary with {Cluster: Subreddit}
        """
        cluster_subreddit_dict = dict()
        for c_num in range(0, self.n_cluster):
            cluster_df = df.loc[df['Cluster_Num'] == c_num]
            most_common_subreddit = None
            subreddit_counts = Counter()

            for row in cluster_df.itertuples():
                subreddit = getattr(row, "Subreddit")
                utils.increment_dict(subreddit, subreddit_counts, 1)
                most_common_subreddit = subreddit_counts.most_common(1)
            cluster_subreddit_dict[c_num] = most_common_subreddit[0][0]

        print(str(cluster_subreddit_dict))
        return cluster_subreddit_dict

    def evaluate_cluster(self, df):
        """
        Purpose: Given some df, containing subreddits and clusternumbers
                 determine what percentage are in the wrong cluster
        Input: df
        Output: percentage of subreddits in the wrong cluster
        """
        cluster_subreddit_dict = self.get_cluster_subreddit(df)
        correct = 0
        total = 0

        for c_num in range(0, self.n_cluster):
            cluster_df = df.loc[df['Cluster_Num'] == c_num]
            correct_subreddit = cluster_subreddit_dict[c_num]

            for row in cluster_df.itertuples():
                subreddit = getattr(row, "Subreddit")
                if subreddit == correct_subreddit:
                    correct += 1
                total += 1

        return correct / total
