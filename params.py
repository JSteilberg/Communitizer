################################################################################
# Purpose: Turns a csv file into a dictionary
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


import csv
from pydoc import locate

def make_csv_dict(file_name):
    """
    Purpose: Converts file specified by file_name into a dictionary
    Input: String containing path to a csv file
    Output: Dictionary
    """
    out_dict = dict()

    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)

        labels = [col.lower() for col in reader.__next__()]
        tipe_idx = labels.index('datatype')
        name_idx = labels.index('name')
        val_idx = labels.index('value')

        for row in reader:
            name = row[name_idx]

            # Get the type for the value and cast
            tipe = locate(row[tipe_idx])

            if tipe == bool:
                value = str2bool(row[val_idx])
            else:
                value = tipe(row[val_idx])

            out_dict[name] = value

    return out_dict


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
