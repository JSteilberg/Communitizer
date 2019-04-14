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

from spherecluster import SphericalKMeans

GRAM_NUM = 1
NUM_CLUSTERS = 2

class Clusternator:
    
    
    def __init__(self, data, vec_model):
        self.data = data
        self.vec_model = vec_model

    

    def run_k_means(self, num_clusters):
        self.skm = SphericalKMeans(n_clusters=num_clusters)
        
        skm.fit(self.data)

        
        

