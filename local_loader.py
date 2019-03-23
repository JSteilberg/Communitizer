################################################################################
# Purpose: Manages the loading of local data for testing the algorithm
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

import json
import random

def sample_file(filename,
                subreddit='all',
                sample_rate=1,
                flip=False,
                min_score=-1e15):
    """
    Purpose: Returns a subset of the given comments based on following parameters
    Input: data      - Data consisting a of a list of comments, which are dictionaries
                       SHOULD BE CLEANED PRIOR TO INPUT
           gram_num  - Size of word-level grams to create. e.g. 1 = unigrams
           sub_name  - Subreddit to sample from, defaults to all reddit
           flip      - If set to True, samples from all BUT subreddit sub_name
           samp_rate - Proportion of total data to sample
           min_score - Toss comments with less than this amount of score
    Returns: List of comments constituting a sample
    """
    data = list()
    file = open(filename, encoding='utf-8', errors='ignore')

    # "error correction"
    subreddit = subreddit.lower()

    # Loop through the file
    for line in file:
        # Loop through each comment in the dataset
        js = json.loads(line)
        if subreddit == 'all' \
           or (js['subreddit'].lower() == subreddit and not flip) \
           or (js['subreddit'].lower() != subreddit and flip):
            # If the comment is from our subreddit, sample at special rate
            if random.random() <= sample_rate:
                data.append(js)

    return data
