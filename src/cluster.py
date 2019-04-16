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
from gensim.models import Word2Vec
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

        print("Getting model...")
        model_filepath = "./models/" + str(self.data_filename + "_model")
        if utils.filepath_exists(model_filepath):
            self.model = Word2Vec.load(model_filepath)
        else:
            model = self.dc.create_model()
            model.save(model_filepath)
            self.model = model

        print("Converting comments to embedding vectors...")
        self.dc.make_comment_embeddings(self.model)


    def run_k_means(self):
        if self.dc is None:
            raise RuntimeError("Must prepare data before running k means")
        print("Clustering comments...")
        data = utils.convert_lol_to_numpy(self.dc.df['Embedded_Comment'])
        self.skm.fit(data)
        self.dc.df['Cluster_Num'] = Series(self.skm.labels_, index=self.dc.df.index)

    def get_clusterwords(self, n_most_common):
        cluster_commonword_dict = dict()
        for c_num in range(0, self.n_cluster):
            cluster_df = self.dc.df.loc[self.dc.df['Cluster_Num'] == c_num]
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

        for subreddit in self.dc.df["Subreddit"].unique():
            corpus_subreddit_counts[subreddit] = len(self.dc.df[self.dc.df["Subreddit"] == subreddit].index)

        for c_num in range(0, self.n_cluster):
            cluster_df = self.dc.df.loc[self.dc.df['Cluster_Num'] == c_num]
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
        """
        d = []
        for c_num in range(0, self.n_cluster):
            cluster_df = self.dc.df.loc[self.dc.df['Cluster_Num'] == c_num]
            cluster_embedding = utils.get_embedding(model, utils.get_top_n_words(cluster_df, n))
            for subreddit in sub_embed_dict:
                subreddit_embedding = sub_embed_dict[subreddit]
                sub_clust_sim = abs(cosine_similarity([cluster_embedding],
                                                      [subreddit_embedding])[0][0])
                d.append((c_num, subreddit, sub_clust_sim))

        res_df = pd.DataFrame(d, columns=('Cluster_Num', 'Subreddit', 'Similarity'))

        return res_df
