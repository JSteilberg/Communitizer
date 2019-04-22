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


from cluster import Clusternator
import pdb
import utils

DATA_FILE = 'test.dat'
DATA_FILE2 = 'RC_2007-06'


def main():
    data = DATA_FILE2
    cnator = Clusternator(data, './cfg/clean_params/clean_params.csv', 5)
    cnator.prepare_data()
    cnator.spherical_k_means()

    cluster_commonword_dict = cnator.get_clusterwords(15)
    utils.write_to_filepath(str(cluster_commonword_dict), "clusterwords.txt")

    cnator.dc.df.to_csv('./clusters.csv')
    print("Evaluating Clusters...")
    cnator.get_cluster_stats()
    pdb.set_trace()


main()

