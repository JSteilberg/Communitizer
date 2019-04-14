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
