# Credit_Risk_Analysis

## Overview of the analysis

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. We will use the credit card, credit dataset from LendingClub, a peer-to-peer lending services company,to predict the credit risk. Therefore, we need to deploy different techniques to train and evaluate models with unbalanced classes. Different libraries and algorithums were used to build and evaluate models using resampling. Using various algorithums the data will be over and undersampled,then, we will use a combinatorial approach of over- and undersampling to fit models for predictions. At the end, we will evaluate the performance of these two new machine learning models.

## Purpose

The main purpose of this analysis is to predict the credit card risk from a credit card usage dataset using imbalanced-learn, scikit-learn libraries and RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN algorithms and to apply machine learning algorithms to solve a real-world challenge in data analytics.

## Resources

- Data source:LoanStats_2019Q1.csv
- Jupyter Notebook 6.4.12

## Results

The results of all six machine learning models including their balanced accuracy score, the precision and recall scores are listed below:

### Naive Random Oversampling

For Naive Random Oversampling, we will use the oversampling RandomOverSampler algorithm to resample the data, and create training and testing groups from the given dataset. The below image shows the respective balanced accuracy score, confusion matrix, and classification report.

![]()

- The balanced accuracy for this model is .615.The precision for the high-risk loans are 0.01 and the precision for low-risk loans are almost 1.00 means correctly predicted.The recall scores for this model evaluate that positive low-risk loans(.63) are slightly higher than high-risk loans(.60).F1 score is a weighted average of the true positive rate (recall) and precision,the F1 score for high-risk loans are .02 and low-risk loans are .78 repectively.

- In summary, this model may not be the best one for preventing fraudulent loans because the model's accuracy, 0.615, is low, and the precision and recall are not good enough to state that the model will be good at classifying fraudulent loans.

### SMOTE Oversampling

The SMOTE Oversampling method used SMOTE algorithm to resample the data, and use the resampled data to train a logistic regression model.

