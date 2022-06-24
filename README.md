# Introduction to Deep Learning for Particle Physics
Hello, 
This repository is for a gentle introduction to deep learning on particle physics data. 

# What is this series for? 
The series consists of four jupyter notebooks and is written in python.
We intend the audience who are interested in deep learning in general and/or physics, specifically particle physics. 
Unfortunately, the data is from CMS, which is not publically available, so you will not be able to run the entire project. However, you could still read and check the results from the notebooks, which is still quite helpful for understanding deep learning used in particle physics. 
In the notebook, we briefly but carefully explained the details of deep learning and each architecture as well as physics behind the data. 
I hope you can find it useful, and this series will be a great introductory path for your physics journey!

# Table of Content
#### Part 1: Overview of CMS and ECAL and the data exploration - `EGamma_ML.ipynb`
#### Part 2: Artificial Neural Network (ANN) - `EGamma_ANN.ipynb`
#### Part 3: Convolutional Neural Network (CNN) and ResNet - `EGamma_CNN.ipynb`
#### Part 4: Summary and Conclusions - `EGamma_summary.ipynb`

# How to run this project? 
1. Clone the repository
2. Open jupyter notebook and run the notebooks

# GPU availability
If you have access to the dataset and want to train the models, I strongly recommend training them on GPU, not CPU as it takes forever. 
If you need access to GPU, please run it through Google CoLab with GPU. 

# What can you learn from this project?
1. Data Science Perspective
  - What is deep learning? How do you encode it?
  - Characteristics of each neural network architecture: ANN, CNN, ResNet
  - How to interpret the results 
  - Common failures in neural network experiments (i.e., overfitting, underfitting)

2. Physics Perspective
-  What is an electromagnetic calorimeter (ECAL)? What does it measure?
-  What is an electromagnetic shower?
-  How do photons and electrons differ?
-  Theoretical background of energy loss of photons and electrons


**This notebook is not for running, but for reading and checking the results as the data is not publically available.**

# Credits
This project is an extension of Kyungmin Park's (Carnegie Mellon University) work https://github.com/kyungminparkdrums/EGamma.git. 
Some tastes were added by Manami Kanemura (Northeastern University), and the entire project was directed by Michael Andrew (Carnegie Mellon University) and Professor Manfred Paulini (Carnegie Mellon University). 


