# Communitizer
Finds and tracks communities on the website reddit using NLP techniques

#### Prerequisites
* Running install_packages.sh should install every necessary file
* A cfg/ directory that contains a clean_params/ directory
    * Should contain both .csv files containing the parameters
* A data/ directory
    * Contains a raw/ directory 
        * Contains RC_2015-06_sub 
        * Contains RC_2013-05
        * Contains RC_2007-02
* The RC_2015-06_sub is a subsample of the RC_2015-06 subreddit
  that contains a small enough set of data that running it will 
  finish in a reasonable amount of time
* Running with larger samples (i.e. 2013) may end up taking 
  around ~1-2 hours
* A models/ directory 

#### 2 Clusters
* With the prerequisites it should be easy to run
this file and get some results
* This file will print out the subreddit make-up of each 
cluster
* It will print out the accuracy
* It will also print out the accuracy of hate comments
  grouped in the correct cluster and non-hate comments
  grouped correctly
* It will create a CSV called 2_clusters that contains some
  comments and their cluster number.
* It will output a 2clusterwords.txt which outputs the top 2
  words in each cluster

#### 6 Clusters
* With the prerequisites it should be easy to run
this file and get some results
* This file will print out the subreddit make-up of each 
cluster
* It will print out the accuracy
* It will create a CSV called 6_clusters that contains some
  comments and their cluster number.
* It will output a 6clusterwords.txt which outputs the top 6
words in each cluster
  
#### 10 Clusters
* With the prerequisites it should be easy to run
this file and get some results
* This file will print out the subreddit make-up of each 
cluster
* It will print out the accuracy
* It will create a CSV called 10_clusters that contains some
  comments and their cluster number.
* It will output a 10clusterwords.txt which outputs the top 10
  words in each cluster
  
* Note that each time this program is run it will create a word2vec
model in the models/ directory