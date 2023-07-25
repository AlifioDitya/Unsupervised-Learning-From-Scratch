# Unsupervised Learning: From Scratch!
This repository features a collection of supervised learning algorithms for classification tasks implemented from scratch in Python. The algorithms are implemented in a Jupyter Notebook format, allowing for easy visualization of the results. A question and answer section is also provided to explain the inner workings of each algorithm, located in the `docs` folder. This is a part of the "**From Scratch!**" series I made as a personal project to further my understanding of machine learning algorithms.

## Algorithms
The following algorithms are implemented:
- DBSCAN
- K-Means
- K-Medoids
The implementation of each algorithm is located in the `src/cluster` folder.

## How to Use
To use the algorithms on your own custom dataset, simply head over to each represented notebooks
located in the `src` folder, and scroll down to the `External Dataset Evaluation` section located all the way on the bottom of the notebook. From there, you can specify the path to your dataset, and the notebook will automatically load the data and evaluate the algorithms on your dataset.  

<img width="500" alt="image" src="https://github.com/AlifioDitya/Unsupervised-Learning-From-Scratch/assets/103266159/6706d7ad-3501-4817-9cd0-9eb15cf36b42">

## Assumptions
The algorithms are implemented with the following assumptions:
- The dataset is in a CSV format
- The dataset contains no missing values and is fully numerical

## Requirements
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Plotly](https://plotly.com/)