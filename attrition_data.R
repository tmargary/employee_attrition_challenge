# removing all objects
rm(list=ls())

# Load dataset into R
attrition_data <- read.csv("attrition_data.csv", 
                           header = TRUE, 
                           stringsAsFactors = FALSE)

# Dropping EMP_ID, JOBCODE, REFERRAL_SOURCE, TERMINATION_YEAR
attrition_drop <- attrition_data[,c(-1, -4, -11, -14)]

# Identifying missing values
summary(attrition_drop)

# Dropping the only missing value
attrition_drop <- attrition_drop[c(-2935), ]

# Log(ANNUAL_RATE)

for(column in c(1, 2, 7))
{
  print('_______________LOG_Transform________________')
  print(names(attrition_drop[column]))
  i <- 1
  while(i <= nrow(attrition_drop)){
    attrition_drop[, column][i] <- log(attrition_drop[, column][i])
    i<-i+1
  }
  
}

# Scaling the dataset
attrition_drop[c(1, 2, 7)] <- scale(attrition_drop[c(1, 2, 7)])
attrition_scaled <- attrition_drop

# Checking if the dataset is ballanced as far as the target var is concerned
summary(attrition_scaled$STATUS) # The dataset is balanced

# Factoring all the columns except for continual var

f_columns <- c(1:length(attrition_scaled))[c(-1, -2, -7, -17)]

for(column in f_columns)
  {
    print('________________Factoring________________')
    print(names(attrition_scaled[column]))
    attrition_scaled[, column] <- factor(attrition_scaled[, column],
                                       order = TRUE,
                                       levels = unique(attrition_scaled[, column]))
    print(summary(attrition_scaled[, column]))
    
}

attrition_cat_final <- attrition_scaled

# Creating scaled numeric data for further analisis and ANN

attrition_numeric_scaled <- attrition_drop

for(column in f_columns)
{
  print('_________________________________')
  print(names(attrition_numeric_scaled[column]))
  attrition_numeric_scaled[, column] <- scale(as.numeric(factor(attrition_numeric_scaled[, column],
                                       order = TRUE,
                                       levels = unique(attrition_numeric_scaled[, column]))))
}

#Setting a random seed
set.seed(123)
idx <- sort(sample(nrow(attrition_cat_final),as.integer(.80*nrow(attrition_cat_final)))) 

# Splitting the dataset in general and for ANN
training_set <- attrition_cat_final[idx,]
test_set <- attrition_cat_final[-idx,]

training_set_nn <- attrition_numeric_scaled[idx,]
test_set_nn <- attrition_numeric_scaled[-idx,]

########################################

# Selecting the features with FORWARD selection

# install.packages('MASS')
# library(MASS)
# FullModel <- glm(as.factor(STATUS) ~., family=binomial, data=training_set)    
# EmptyModel <- update(FullModel, . ~ 1)

# forwards = step(EmptyModel, scope=list(lower=.~1, upper=formula(FullModel)), direction="forward")

# JOB_GROUP + ANNUAL_RATE + PREVYR_1 + PREVYR_5 + PREVYR_3 + PREVYR_4

# backwards = step(FullModel) 
# formula(backwards)  

# JOB_GROUP + ANNUAL_RATE + PREVYR_1 + PREVYR_5 + PREVYR_3 + PREVYR_4

# stepwise = step(EmptyModel, scope=list(lower=.~1, upper=formula(FullModel)), direction="both")

# JOB_GROUP + ANNUAL_RATE + PREVYR_5 + PREVYR_1 + PREVYR_4 + PREVYR_3 + TRAVELLED_REQUIRED

########################################

var <- c('ANNUAL_RATE', 'JOB_GROUP', 'PREVYR_1', 'PREVYR_3', 'PREVYR_4', 'PREVYR_5', 'STATUS')
predictors <- var[-7]

training_set <- training_set[var]
test_set <- test_set[var]

########################################

# Fitting the model
model_lr = glm(formula = as.factor(STATUS) ~., 
               family = binomial, 
               data = training_set)
summary(model_lr)
car::vif(model_lr)

# Predicting the Test set results
prob_pred = predict(model_lr, 
                    type = 'response', 
                    newdata = test_set[,predictors])
pred_lr = ifelse(prob_pred > 0.5, 'T', 'A')

# Making the Confusion Matrix
table(test_set$STATUS, as.factor(pred_lr))

# Accuracy
lr_wrong <- sum(pred_lr != test_set$STATUS)
acc_lr <- (1 - lr_wrong / length(pred_lr)) * 100
acc_lr

########################################

# install.packages('e1071')
library(e1071)

# Fitting the model
model_nb <- naiveBayes(as.factor(STATUS) ~., 
                       data = training_set)

# Predicting the Test set results
pred_nb <- predict(model_nb, 
                   newdata = test_set[,predictors])


# Making the Confusion Matrix
table(test_set$STATUS, as.factor(pred_nb))

# Accuracy
NB_wrong <- sum(pred_nb != test_set$STATUS)
acc_nb <- (1 - NB_wrong / length(pred_nb)) * 100
acc_nb

########################################

library(kknn) 

for(i in c(1, 3, 5, 7)){
  # Fitting the model
  model_knn <- kknn(formula=as.factor(STATUS)~., 
                    training_set, 
                    test_set[,predictors], 
                    k=i,
                    kernel ="rectangular"  )
  
  pred_knn <- fitted(model_knn)
  
  # Accuracy
  
  knn_wrong <- sum(pred_knn != test_set$STATUS)
  acc_rate<-(1 - knn_wrong / length(pred_knn)) * 100
  
  print('***************')
  print(i)
  print( table(test_set$STATUS,pred_knn))
  print( acc_rate)
  print('***************') 
}

# final knn
# Fitting the model
model_knn <- kknn(formula=as.factor(STATUS)~., 
                  training_set, 
                  test_set[,predictors], 
                  k=7,
                  kernel ="rectangular"  )
pred_knn <- fitted(model_knn)

knn_wrong <- sum(pred_knn != test_set$STATUS)
acc_knn <- (1 - knn_wrong / length(pred_knn)) * 100
acc_knn

########################################

# install.packages("rpart")
# install.packages("rpart.plot")     # Enhanced tree plots
# install.packages("rattle")         # Fancy tree plot
# install.packages("RColorBrewer")   # colors needed for rattle
library(rpart)
library(rpart.plot)  			# Enhanced tree plots
library(rattle)           # Fancy tree plot
library(RColorBrewer)     # colors needed for rattle

# install.packages('rpart')
library(rpart)

# Fitting the model
model_dt = rpart(formula = as.factor(STATUS) ~ .,
                   data = training_set)

# Graphs
rpart.plot(model_dt)
prp(model_dt)
# Much fancier graph
fancyRpartPlot(model_dt)

# Predicting the Test set results
pred_dt <- predict(model_dt, 
                   newdata = test_set[,predictors], 
                   type = 'class')

# Making the Confusion Matrix
table(test_set$STATUS, pred_dt)

# Accuracy
dt_wrong <- sum(pred_dt != test_set$STATUS)
acc_dt <- (1 - dt_wrong / length(pred_dt)) * 100
acc_dt

########################################

# install.packages("randomForest")
library(randomForest)

# Fitting the model
model_rf <- randomForest(as.factor(STATUS) ~ ., 
                         data = training_set, 
                         ntree = 500, 
                         mtry = 6, 
                         importance = TRUE)
importance(model_rf)
varImpPlot(model_rf)

# Predicting the Test set results
pred_rf <- predict(model_rf, 
                   newdata = test_set[,predictors], 
                   type = 'class')

# Making the Confusion Matrix
table(test_set$STATUS, pred_rf)

# Accuracy
rf_wrong <- sum(pred_rf != test_set$STATUS)
acc_rf <- (1 - rf_wrong / length(pred_rf)) * 100
acc_rf

########################################

# Fitting SVM to the Training set 
# install.packages('e1071') 
library(e1071) 

# Fitting the model
model_svm = svm(formula = as.factor(STATUS) ~ ., 
                 data = training_set, 
                 type = 'C-classification', 
                 kernel = 'linear', 
                 probability=TRUE) 

# Predicting the Test set results 
pred_svm = predict(model_svm, 
                   newdata = test_set[,predictors], 
                   probability=TRUE) 

# Making the Confusion Matrix 
table(test_set$STATUS, pred_svm)

# Accuracy
svm_wrong <- sum(pred_svm != test_set$STATUS)
acc_svm <- (1 - svm_wrong / length(pred_svm)) * 100
acc_svm

########################################

library(e1071)

# Fitting the model
model_svm_radial = svm(formula = as.factor(STATUS) ~ ., 
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial', 
                 probability=TRUE)

# Predicting the Test set results
pred_svm_radial = predict(model_svm_radial, 
                          newdata = test_set[,predictors], 
                          probability=TRUE)

# Making the Confusion Matrix
table(test_set$STATUS, pred_svm_radial)

# Accuracy
svm_radial_wrong <- sum(pred_svm_radial != test_set$STATUS)
acc_svm_radial <- (1 - svm_radial_wrong / length(pred_svm_radial)) * 100
acc_svm_radial

########################################

library(neuralnet)

# Fitting the model
model_ann <- neuralnet(formula = as.factor(STATUS) ~ .,
                       training_set_nn,
                       hidden=7,
                       linear.output=FALSE)

# plot neural network
plot(model_ann)

# Predicting the Test set results
pred_ann_prob <- neuralnet::compute(model_ann,test_set_nn[-17])
pred_ann <- sapply(pred_ann_prob$net.result[,1],round,digits=0)
pred_ann <- as.factor(ifelse(pred_ann == 1, 'A', 'T'))

# Making the Confusion Matrix 
table(test_set_nn$STATUS, pred_ann)

# Accuracy
ann_wrong <- sum(pred_ann != test_set_nn$STATUS)
acc_ann <- (1 - ann_wrong / length(pred_ann)) * 100
acc_ann

########################################

#install.packages("C50", repos="http://R-Forge.R-project.org")
#install.packages("C50")
 
library('C50')

# Fitting the model
model_cs50 <- C5.0(as.factor(STATUS)~.,
                   data=training_set)
summary(model_cs50)
#plot(model_cs50)

# Predicting the Test set results
pred_c50 <- predict(model_cs50 ,test_set[,predictors])

# Making the Confusion Matrix
table(test_set$STATUS, pred_c50)

# Accuracy
c50_wrong <- sum(pred_c50 != test_set$STATUS)
acc_c50 <- (1 - c50_wrong / length(pred_c50)) * 100
acc_c50

########## ensemble model ##############

test_set$pred_lr_prob <- 3 * (predict(object = model_lr,test_set[,predictors],type='response'))
#test_set$pred_nb_prob <- 1-predict(object = model_nb,test_set[,predictors],type='raw')
test_set$pred_knn_prob <- 3 * (1-predict(object = model_knn,test_set[,predictors],type='prob'))
test_set$pred_dt_prob <- 1-predict(object = model_dt,test_set[,predictors],type='prob')
test_set$pred_rf_prob <- 1-predict(object = model_rf,test_set[,predictors],type='prob')
test_set$pred_svm_prob <- attr(pred_svm, "probabilities")
test_set$pred_svm_radial_prob <- attr(pred_svm_radial, "probabilities")
#test_set$pred_ann <- 1-pred_ann_prob$net.result[,1]
test_set$pred_c50 <- 10 * (1-predict(object = model_cs50 ,test_set[,predictors],type='prob'))


#Taking average of predictions

test_set$pred_avg_prob <-(test_set$pred_lr_prob+
                            #test_set$pred_nb_prob+
                            test_set$pred_knn_prob+
                            test_set$pred_dt_prob+
                            test_set$pred_rf_prob+
                            test_set$pred_svm_prob+
                            test_set$pred_svm_radial_prob+
                            #test_set$pred_ann+
                            test_set$pred_c50)/20
test_set$pred_avg_cat <- as.data.frame(ifelse(test_set$pred_avg_prob > 0.5, 'T', 'A'))$A

# Making the Confusion Matrix 
table(test_set$STATUS, test_set$pred_avg_cat)

# Accuracy
ENS_wrong <- sum(test_set$pred_avg_cat != test_set$STATUS)
acc_ens <- (1 - ENS_wrong / length(test_set$pred_avg_cat)) * 100 

# All the accuracy rates
algorithm <- c('Multivariate Logistic Regression', 
               'Naive Bayes', 
               'K-Nearest Neighbor', 
               'Decision Tree', 
               'Random Forest', 
               'Support Vector Machines (SVM) with Linear Kernel', 
               'Support Vector Machines (SVM) with Radial Kernel', 
               'Artificial Neural Network', 
               'C5.0', 
               'Ensemble Model')
algorithm_acc <- c(acc_lr, acc_nb, acc_knn, acc_dt, acc_rf, acc_svm, acc_svm_radial, acc_ann, acc_c50, acc_ens)
acc <- data.frame(algorithm, algorithm_acc)
View(acc)