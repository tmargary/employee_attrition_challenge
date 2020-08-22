# Employee Attrition Challenge by K. Dehnad (Stevens Institute of Technology)
Final Project for CS 513 Knowledge Discovery and Data Mining
The dataset was provided as a challenge by professor K. Dehnad.

## Problem Statement:
To develop a classification model(s) to predict the potential of employees leaving the company. 

## Data Preparation:
In this stage, `EMP_ID`, `JOBCODE`, `REFERRAL_SOURCE`, `TERMINATION_YEAR` columns are also dropped as the
first two are random numbers, and `TERMINATION_YEAR` is a retrospectively collected data. In other words, If we are to predict the STATUS of current employees, TERMINATION_YEAR should not be included in the model. 
`REFERRAL_SOURCE` is dropped too because, as the analysis shows (forwards selection), it does not have significance, and there is no need to treat the missing values here. As the rest of the dataset is clean, and there are not many null values, the raws are dropped.

As the dataset doesn’t have any missing values, I have performed log transformation for the continual variables as it results in slightly better residuals. Next, I have scaled them.
In our next step, we have checked if our target variable `STATUS` column is balanced. Since the column is almost perfectly balanced (1 – 5394, 0 – 4217), I have factored the categorical columns using the for loop, as well as processed the data for ANN (fully numerical dataset).

## Modeling:
For variable selection, I have performed forward selection, and the results are as follows:
`Forwards: JOB_GROUP + ANNUAL_RATE + PREVYR_1 + PREVYR_5 + PREVYR_3 + PREVYR_4`
We have also performed backward and stepwise selection, and the result was almost the same.
`Backwards: JOB_GROUP + ANNUAL_RATE + PREVYR_1 + PREVYR_5 + PREVYR_3 + PREVYR_4`
`Stepwise: JOB_GROUP + ANNUAL_RATE + PREVYR_5 + PREVYR_1 + PREVYR_4 + PREVYR_3 + TRAVELLED_REQUIRED`

As our dataset is big enough, I have done 80/20 split. The training and test sets of ANN include all the preprocessed columns (in hopes of detecting the anomalies in the data). In contrast, training and test sets of other models include only the significant columns determined by forwards selection.

These are the algorithms that I have used for our predictions: 

- Multivariate Logistic Regression, 
- Naive Bayes, 
- K-Nearest Neighbor, 
- Decision Tree, 
- Random Forest,
- Support Vector Machines (SVM) with Linear Kernel,
- Support Vector Machines (SVM) with Radial Kernel,
- Artificial Neural Network, 
- C5.0, 
- Ensemble Model.

In the Ensemble Model, I have calculated the weighted average of the predicted probabilities of the most successful algorithms that we have used during the analysis.
Ensemble Model gave the highest accuracy of 75.14%.
