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

![](https://github.com/akthersr/Credit_Risk_Analysis/blob/main/Resources/Naive.png)

- The balanced accuracy for this model is around 61%.The precision for the high-risk loans are 0.01 and the precision for low-risk loans are almost 1.00 means correctly predicted.The recall scores for this model evaluate that positive low-risk loans(.63) are slightly higher than high-risk loans(.60).F1 score is a weighted average of the true positive rate (recall) and precision,the F1 score for high-risk loans are .02 and low-risk loans are .78 repectively.

- In summary, this model may not be the best one for preventing fraudulent loans because the model's accuracy, 0.615, is low, and the precision and recall are not good enough to state that the model will be good at classifying fraudulent loans.

### SMOTE Oversampling

In SMOTE Oversampling method, we used SMOTE algorithm to resample the data, and use the resampled data to train a logistic regression model. The below image shows the respective balanced accuracy score, confusion matrix, and classification report.

![](https://github.com/akthersr/Credit_Risk_Analysis/blob/main/Resources/smote.png)

- The balanced accuracy score for this model is around 62%,so, the model predicted credit risk accurately.The precision for the high-risk loans are 0.01 and the precision for low-risk loans are almost 1.00 means correctly predicted.The recall scores for this model evaluate that positive low-risk loans(.65) are slightly higher than high-risk loans(.60).So, this model is not good for predicting high-risk loans.

- The F1 score for this model are similar to Naive Random Oversampling method. The F1 score for high-risk loans are .02 and low-risk loans are .78 repectively. This is a good model for predicting low-risk loans than high-risk loans.

### Undersampling

 For Undersampling, we will use the Cluster Centroids algorithm to resample the data, and create training and testing groups from the given dataset. The below image shows the respective balanced accuracy score, confusion matrix, and classification report.

![](https://github.com/akthersr/Credit_Risk_Analysis/blob/main/Resources/cluster.png)

- The balanced accuracy score for Undersampling model is around 52%,means the model predicted the lowest credit risk of all the models. So, about 52% of all testing data was classified properly.
- The precision score for this model are positively skewed towards low-risk loans, which is 1.00. But, for high-risk loans the score is minimal 0.01,means this model is not a good fit for high-risk loans.
- The recall score for high-risk and low-risk loans are 60% and 43% respectively.The F1 score for high-risk and low-risk loans are .01 and .60 respectively.We can predict that this model is not great for identifying high-risk loans.

### Combination (Over and Under) Sampling

In this method we will resample the data using the SMOTEENN algorithm to resample the data.The logistic regression model was fitted to get the respective balanced accuracy score, confusion matrix, and classification report.

![](https://github.com/akthersr/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN.png)

- The balanced accuracy score for Undersampling model is around 53%, means the model predicted the lowest credit risk of all the models. So, about 53% of all testing data was classified properly.
- The precision score for this model are positively skewed towards low-risk loans, which is 1.00. But, for high-risk loans the score is minimal 0.01,means this model is not a good fit for high-risk loans.
- The recall score for high-risk and low-risk loans are 72% and 58% respectively.In comparision to other methods,this model is good at identifying high-risk loans.
- The F1 score for high-risk and low-risk loans are .02 and .73 respectively.We can predict that this model is not great for identifying high-risk loans.

### Balanced Random Forest Classifier

In Balanced Random Forest Classifier method, we used Balanced Random Forest Classifier algorithum to resample the training data with 100 estimators to classify the testing data. The below image shows the balanced accuracy score, confusion matrix, and classification report respectively.

![](https://github.com/akthersr/Credit_Risk_Analysis/blob/main/Resources/random%20forest.png)

- The balanced accuracy score for this model is higher than other which is almost 78%, so, the the testing data was 78% accurately classified.
- The precision score for high-risk loans is 0.03 which is very low compare to low-risk loans 1.00, indicates may be a large number of false negatives.
- The recall score for low-risk loans are vary high almost 89% in comparision with high-risk loans 68%, indicates that the classifier can predict true positives for low-risk loans.
- The F1 score for low-risk loans are 94%, indicates the model is a good fit for classifying low-risk loan.

### Easy Ensemble AdaBoost Classifier

![](https://github.com/akthersr/Credit_Risk_Analysis/blob/main/Resources/ADA.png)
