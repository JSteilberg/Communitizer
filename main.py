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

from data_clean import DataCleaner
from cluster import Clusternator


cleaner = DataCleaner('./data/raw/RC_2007-02', './cfg/clean_params/clean_params.csv')

cleaner.load_data_for_word2vec()

print("Creating model...")
model = cleaner.create_model()
model.save("./models/" + str(datetime.now()).replace(':', '.') + "_model")

print("Converting comments to embedding vectors...")
embeds = cleaner.make_comment_embeddings(model)

print("Clustering comments...")
cnator = Clusternator(embeds)
skm = cnator.run_k_means(3)

import pdb
pdb.set_trace()
