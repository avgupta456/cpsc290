# About

Project developed during Spring 2020 for CPSC 290: Directed Research in Computer Science at Yale University, under the mentorship of Dr. Vazquez. 

Implemented the DANTE group detection algorithm, as described here: https://arxiv.org/pdf/1907.12910.pdf. Essentially, using deep learning to predict conversation groupings from the locations of a set of individuals. I made improvements to robustness and scalability of model (handling variable input size, improved speed, etc).

# How to Implement

Please use python 3.7 and install the requirements.txt file. In the 'new' folder, run 'process.py' to process raw input and generate files for training the deep learning algorithm. Use 'run_model.py' to train and evaluate models on each fold. Update constants in 'constants.py' to change dataset, modify other parameters.


