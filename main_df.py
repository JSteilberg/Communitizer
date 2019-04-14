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


from datetime import datetime
from pandas import Series
from data_clean import DataCleaner
from data_clean_df import DataCleanerDF
from cluster import Clusternator
from gensim.models import Word2Vec
import pdb
import utils

DATA_FILE = 'test.dat'
DATA_FILE2 = 'RC_2007-02'


def main():
    data = DATA_FILE2
    cleaner = DataCleanerDF('./data/raw/' + data, './cfg/clean_params/clean_params.csv')

    cleaner.load_data_for_word2vec()

    print("Getting model...")
    model_filepath = "./models/" + str(data + "_model")
    if utils.filepath_exists(model_filepath):
        model = Word2Vec.load(model_filepath)
    else:
        model = cleaner.create_model()
        model.save(model_filepath)

    df = cleaner.df

    print("Converting comments to embedding vectors...")
    embeds = cleaner.make_comment_embeddings(model)

    print("Clustering comments...")
    cnator = Clusternator(embeds)
    skm = cnator.run_k_means(10)

    df['Cluster Num'] = Series(skm.labels_, index=df.index)

    df.to_csv('./clusters.csv')


main()
