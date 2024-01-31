# Predicting Student Examination Score

## Table of Contents
  - Project Overview

  - Tools

  - Data Cleaning

  - Supervised Machine Learning

  - Insights

### Project Overview
This Supervised Machine learning data science project aims to generate insights for creating a model that effectively predicts a student examination score. Through the analysis of the dataset, the goal is to identify a model that accurately predicts how well a student is likely to perform in his/her examination based on the hours put into studying for that exam.

### Tools
 - Python - Data Analysis
 - Matplotlib
 - Numpy
 - Pandas
 - Scikit-learn
 - Jupyter notebook - Coding Environment 

### Data Cleaning
In the data preparation phase, i performed the following tasks;

1. Data loading
2. Data Validation
3. Data cleaning

### Supervised Machine Learning
Supervised machine learning is a type of machine learning where the algorithm is trained on a labeled dataset, meaning that the input data is paired with corresponding output labels. The goal of supervised learning is to learn a mapping or relationship between input features and their corresponding output labels, allowing the algorithm to make predictions or classifications on new, unseen data. . Linear Regression was used in creating a model for the data set.

Steps carried out:
   1. I created a scatter plot to visualize the relationship between study hours and examination score.
   2. I checked for the correlation between both variable using the numpy.corrcoef function.
   3. I created a train and test dataset using scikit learn train_test_split function.
   4. I created a  Linear Regression model and fitted it to the train dataset.
   5. I evaluted the performance of the model , by checking its performance accuracy on the test dataset using the MeanSquaredError(MSE) to calculate the RootMeanSquareError(RMSE).
   
### Insights
1.There is a linear relationship between study hours and examination score.

2. Both variables are strongly positive correlated (correlation score of 0.97)
   
3.The model had a R-squared score of 0.94 (94%) and a intercept of 1.94.

4.The RMSE score was 20.46

5.From the model it was predicted that a student who studies for 9hours 25mins will have a score of 93.87 for his or her exams.
This note contains code cells and well documented step by step explanation on using Linear regression in predicting the score of a student by their amount of study time.
