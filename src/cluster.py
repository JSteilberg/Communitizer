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
from collections import Counter
from spherecluster import SphericalKMeans
import utils
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data_clean_df import DataCleanerDF
from pandas import Series


class Clusternator:

    def __init__(self, data_filename, params_loc, num_clusters):
        self.data_filename = data_filename
        self.params_loc = params_loc
        self.n_cluster = num_clusters
        self.skm = SphericalKMeans(n_clusters=self.n_cluster)

        self.model = None
        self.dc = None

    def prepare_data(self):
        self.dc = DataCleanerDF('./data/raw/' + self.data_filename,
                                self.params_loc)
        self.dc.load_data_for_word2vec()

        print("Creating new model")
        model_filepath = "./models/" + str(self.data_filename + "_model")
        model = self.dc.create_model()
        model.save(model_filepath)
        self.model = model

        print("Converting comments to embedding vectors...")
        self.dc.make_comment_embeddings(self.model)

    def spherical_k_means(self):
        if self.dc is None:
            raise RuntimeError("Must prepare data before running k means")
        print("Clustering training comments...")
        data = utils.convert_lol_to_numpy(self.dc.training_embedded_comments)
        self.skm.fit(data)
        self.dc.training_df['Cluster_Num'] = Series(self.skm.labels_, index=self.dc.training_df.index)
        test_data = utils.convert_lol_to_numpy(self.dc.testing_embedded_comments)
        test_labels = self.skm.predict(test_data)
        self.dc.test_df['Cluster_Num'] = test_labels

    def get_clusterwords(self, n_most_common):
        cluster_commonword_dict = dict()
        for c_num in range(0, self.n_cluster):
            cluster_df = self.dc.training_df.loc[self.dc.training_df['Cluster_Num'] == c_num]
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

    def get_cluster_stats(self):
        """
        Purpose: Given some df it determines the percentage of (all of
        the specific subreddits in the corpus) that a given cluster has.
        i.e. if Programming has 30 comments total in the corpus
             cluster A has 10 programming comments => programming: 0.33
             cluster B has 20 programming comments => programming: 0.66

        Input:  DF
        Output: Dictionary with {Cluster: {size:   n
                                           Subreddit1: percentage
                                           Subreddit2: percentage
                                           ...}}
        """
        cluster_subreddit_dict = dict()
        corpus_subreddit_counts = dict()

        for subreddit in self.dc.training_df["Subreddit"].unique():
            corpus_subreddit_counts[subreddit] = len(self.dc.training_df[self.dc.training_df["Subreddit"] == subreddit].index)

        for c_num in range(0, self.n_cluster):
            cluster_df = self.dc.training_df.loc[self.dc.training_df['Cluster_Num'] == c_num]
            subreddit_counts = Counter()

            for row in cluster_df.itertuples():
                subreddit = getattr(row, "Subreddit")
                utils.increment_dict(subreddit, subreddit_counts, 1)

            cluster_stats = dict({
                'cluster_post_count': len(cluster_df.index)
            })

            for key, value in subreddit_counts.items():
                cluster_stats[key] = value / corpus_subreddit_counts[key]

            cluster_subreddit_dict[c_num] = cluster_stats

        print(str(cluster_subreddit_dict))
        return cluster_subreddit_dict

    def get_subreddit_similarity(self, sub_embed_dict, model, n):
        """
        Purpose: Given a df containing some comments in particular clusters,
                 this method does a cosine similarity between each cluster and
                 each subreddit, returning a df that compares a subreddit to a
                 cluster and contains cosine similarity.
        :param df: A dataframe containing comments and their clusters.
        :param sub_embed_dict: A dictionary with {subreddit: embedding vector}
        :param model: The word2vec model to embed each cluster with.
        :param n: The number of words to consider per df
        :return: A DF with (Cluster, Subreddit, Cosine Similarity)
        :return: An array with the most similar subreddit per cluster
        """
        d = []
        cluster_subreddit_labels = []
        for c_num in range(0, self.n_cluster):
            cluster_df = self.dc.training_df.loc[self.dc.training_df['Cluster_Num'] == c_num]
            cluster_embedding = utils.get_embedding(model, utils.get_top_n_words(cluster_df, n))
            max_sim = -float('inf')
            cluster_subreddit = None

            for subreddit in sub_embed_dict:
                if subreddit != 'all':
                    subreddit_embedding = sub_embed_dict[subreddit]
                    sub_clust_sim = abs(cosine_similarity([cluster_embedding],
                                                          [subreddit_embedding])[0][0])
                    if sub_clust_sim > max_sim:
                        max_sim = sub_clust_sim
                        cluster_subreddit = subreddit
                    d.append((c_num, subreddit, sub_clust_sim))

            cluster_subreddit_labels.append(cluster_subreddit)

        res_df = pd.DataFrame(d, columns=('Cluster_Num', 'Subreddit', 'Similarity'))

        return res_df, cluster_subreddit_labels

    def evaluate_cluster(self, cluster_subreddit_labels):
        """
        Determine the max avg similar subreddit to this cluster.
        Count the number of correctly clustered subreddits, and divide
        by the total number of subreddits.
        :param cluster_subreddit_labels: A list of subreddit labels per cluster
        :return: A percentage correct
        """
        correct = 0
        for c_num in range(0, self.n_cluster):
            cluster_df = self.dc.test_df.loc[self.dc.test_df['Cluster_Num'] == c_num]
            cluster_subreddit = cluster_subreddit_labels[c_num]
            for row in cluster_df.itertuples():
                row_subreddit = getattr(row, "Subreddit")

                if row_subreddit == cluster_subreddit:
                    correct += 1
        return correct / len(self.dc.test_df.index)

    def evaluate_hate_cluster(self, cluster_subreddit_labels, hate_subreddit):
        correct = 0
        hate_clustered_correctly = 0
        hate_clustered_incorrectly = 0
        total_hate_cluster = 0

        non_hate_clustered_correctly = 0
        non_hate_clustered_incorrectly = 0
        total_non_hate_cluster = 0

        for c_num in range(0, self.n_cluster):
            cluster_df = self.dc.test_df.loc[self.dc.test_df['Cluster_Num'] == c_num]
            cluster_subreddit = cluster_subreddit_labels[c_num]
            if cluster_subreddit == hate_subreddit:
                for row in cluster_df.itertuples():
                    row_subreddit = getattr(row, "Subreddit")

                    if row_subreddit == hate_subreddit:
                        correct += 1
                        hate_clustered_correctly += 1
                    else:
                        non_hate_clustered_incorrectly += 1
                    total_hate_cluster += 1
            else:
                for row in cluster_df.itertuples():
                    row_subreddit = getattr(row, "Subreddit")

                    if row_subreddit != hate_subreddit:
                        correct += 1
                        non_hate_clustered_correctly += 1
                    else:
                        hate_clustered_incorrectly += 1
                    total_non_hate_cluster += 1

        total_acc = correct / len(self.dc.test_df.index)
        hate_correct_percent = hate_clustered_correctly / total_hate_cluster
        non_hate_correct_percent = non_hate_clustered_correctly / total_non_hate_cluster

        return total_acc, hate_correct_percent, non_hate_correct_percent
