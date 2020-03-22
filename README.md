# Data Matching with Active Machine Learning

[View the notebook here](https://nbviewer.jupyter.org/urls/bitbucket.org/marshackVB/projects/raw/ee0a69382a75226235e41a65b8e13bd9fa678780/data_matching/Data%20Matching.ipynb)  

This project uses several advanced techniques to solve a data matching problem. It was inspired by [a blog post](https://www.enigma.com/blog/on-wages-and-hygiene-surfacing-bad-management-in-public-data). For this project, I am focused exclusively on matching records that refer to the same entity and with as little effort as possible.

Data matching problems follow the same general set of steps though there are many ways these steps can be implemented. A good solution to one problem may not generalize to another problem. To complicate things further, training data will likely not be available. Even if some labeled matches and non-matches are available, it is unlikely that they will be representative of all the available patterns in the data.


I've tried to make my solution more general. There are two key components. An approximate similarity join is used to group match-candidate record pairs for more rigorous comparison. This drastically reduces the number of comparisons to only pairs above a baseline similarity threshold. Then, active machine learning is used to efficiently label a sample of these candidate record pairs to train a machine learning model. Lastly, a match probability is assigned to each pair. Model performance is assesses via precision and recall.
