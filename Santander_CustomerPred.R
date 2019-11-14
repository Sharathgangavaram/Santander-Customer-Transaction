#Remove all existing environment
rm(list=ls(all=T))
#set the working directory
setwd("C:/Data science/Project/Santander Customer Transaction")
#check the working directory
getwd()
#load libraries
X=c("ggplot2","corrgram","DMwR","caret","randomForest","unbalanced","C50","dummies","MASS","rpart",
    "gbm","ROSE","e1071","Information","sampling","DataCombine","inTrees","readxl","e1071","tidyverse","moments",
    "DataExplorer","Matrix","pdp","mlbench","caTools","glmnet","mlr","rBayesianOptimization","lightgbm","pROC",
    "yardstick")

#Installing packages
install.packages(c("randomForest","unbalanced","c50","dummies","MASS","rpart","gbm","ROSE","e1071","Information",
                   "e1071","tidyverse","moments","DataExplorer","Matrix","pdp","mlbench","caTools","glmnet","mlr",
                   "rBayesianOptimization","lightgbm","pROC","yardstick","DataCombine"))

lapply(X,require,character.only=TRUE)

rm(X)

#Creating Sample data from Large Data
sample_size = 10000
set.seed(1)
idxs = sample(1:nrow(Cus_test),sample_size,replace=F)
subsample_test = Cus_test[idxs,]
pvalues = list()
for (col in names(Cus_test)) {
  if (class(Cus_test[,col]) %in% c("numeric","integer")) {
    # Numeric variable. Using Kolmogorov-Smirnov test
    
    pvalues[[col]] = ks.test(subsample_test[[col]],Cus_test[[col]])$p.value
    
  } else {
    # Categorical variable. Using Pearson's Chi-square test
    
    probs = table(Cus_test[[col]])/nrow(Cus_test)
    pvalues[[col]] = chisq.test(table(subsample_test[[col]]),p=probs)$p.value
    
  }
}
pvalues1=pvalues>=0.4
sum(pvalues1==FALSE)

write.csv(subsample,'Santander_sample_train.csv',row.names = F)
write.csv(subsample_test,'Santander_sample_test.csv',row.names = F)

#Loading Data set
#Cus_train=read.csv("train.csv",header = TRUE,na.strings = c(" ", "", "NA"))
Cus_train=read.csv("Santander_sample_train.csv",header = TRUE,na.strings = c(" ", "", "NA"))
# Loading test data
#Cus_test=read.csv("test.csv",header = TRUE)
Cus_test=read.csv("Santander_sample_test.csv",header = TRUE)
#####Explorataory Data Analysis##########
str(Cus_train)
str(Cus_test)
head(Cus_train)
head(Cus_test)
summary(Cus_train)
summary(Cus_test)
#Let's see the percentage of our target variable
round(prop.table(table(Cus_train$target))*100,2)
#Bar plot for count of target classes
plot1<-ggplot(Cus_train,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightgreen')
plot1
#Distribution of train attributes from 3 to 102
for (var in names(Cus_train)[c(3:102)]){
  target<-Cus_train$target
  plot<-ggplot(Cus_train, aes(x=Cus_train[[var]],fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}

#Distribution of train attributes from 103 to 202

for (var in names(Cus_train)[c(103:202)]){
  target<-Cus_train$target
  plot<-ggplot(Cus_train, aes(x=Cus_train[[var]],fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}
#We can observed that their is a considerable number of features which are significantly have different distributions
#for two target variables. For example like var_0,var_1,var_9,var_198 var_180 etc.
#We can observed that their is a considerable number of features which are significantly have same distributions
#for two target variables. For example like var_3,var_7,var_10,var_171,var_185 etc.

###Let us see distribution of test attributes from 2 to 101

plot_density(Cus_test[,c(2:101)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))

#Distribution of test attributes from 102 to 201
plot_density(Cus_test[,c(102:201)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))

#We can observed that their is a considerable number of features which are significantly have different
#distributions. For example like var_0,var_1,var_9,var_180 var_198 etc.
#We can observed that their is a considerable number of features which are significantly have same distributions.
#For example like var_3,var_7,var_10,var_171,var_185,var_192 etc.

#############Let us see distribution of mean values per row and column in train and test dataset########

#Applying the function to find mean values per row in train and test data.
train_mean<-apply(Cus_train[,-c(1,2)],MARGIN=1,FUN=mean)
test_mean<-apply(Cus_test[,-c(1)],MARGIN=1,FUN=mean)
ggplot()+
  #Distribution of mean values per row in train data
  geom_density(data=Cus_train[,-c(1,2)],aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=Cus_test[,-c(1)],aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per row',title="Distribution of mean values per row in train and test dataset")

#Applying the function to find mean values per column in train and test data.
train_mean<-apply(Cus_train[,-c(1,2)],MARGIN=2,FUN=mean)
test_mean<-apply(Cus_test[,-c(1)],MARGIN=2,FUN=mean)
ggplot()+
  #Distribution of mean values per column in train data
  geom_density(aes(x=train_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per column in test data
  geom_density(aes(x=test_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per column',title="Distribution of mean values per row in train and test dataset")

#########Let us see distribution of standard deviation values per row and column in train and test dataset####

#Applying the function to find standard deviation values per row in train and test data.
train_sd<-apply(Cus_train[,-c(1,2)],MARGIN=1,FUN=sd)
test_sd<-apply(Cus_test[,-c(1)],MARGIN=1,FUN=sd)
ggplot()+
  #Distribution of sd values per row in train data
  geom_density(data=Cus_train[,-c(1,2)],aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=Cus_test[,-c(1)],aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per row',title="Distribution of sd values per row in train and test dataset")

#Applying the function to find sd values per column in train and test data.
train_sd<-apply(Cus_train[,-c(1,2)],MARGIN=2,FUN=sd)
test_sd<-apply(Cus_test[,-c(1)],MARGIN=2,FUN=sd)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per column',title="Distribution of std values per column in train and test dataset")

###########Let us see distribution of skewness values per row and column in train and test dataset####
#Applying the function to find skewness values per row in train and test data.
train_skew<-apply(Cus_train[,-c(1,2)],MARGIN=1,FUN=skewness)
test_skew<-apply(Cus_test[,-c(1)],MARGIN=1,FUN=skewness)
ggplot()+
  #Distribution of skewness values per row in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per row',title="Distribution of skewness values per row in train and test dataset")

#Applying the function to find skewness values per column in train and test data.
train_skew<-apply(Cus_train[,-c(1,2)],MARGIN=2,FUN=skewness)
test_skew<-apply(Cus_test[,-c(1)],MARGIN=2,FUN=skewness)
ggplot()+
  #Distribution of skewness values per column in train data
  geom_density(aes(x=train_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=test_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per column',title="Distribution of skewness values per column in train and test dataset")

###########Let us see distribution of kurtosis values per row and column in train and test dataset#
#Applying the function to find kurtosis values per row in train and test data.
train_kurtosis<-apply(Cus_train[,-c(1,2)],MARGIN=1,FUN=kurtosis)
test_kurtosis<-apply(Cus_test[,-c(1)],MARGIN=1,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per row',title="Distribution of kurtosis values per row in train and test dataset")

#Applying the function to find kurtosis values per column in train and test data.
train_kurtosis<-apply(Cus_train[,-c(1,2)],MARGIN=2,FUN=kurtosis)
test_kurtosis<-apply(Cus_test[,-c(1)],MARGIN=2,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=train_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=test_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per column',title="Distribution of kurtosis values per column in train and test dataset")

#####Let us do Missing value analysis###
#Finding the missing values in train data
missing_val<-data.frame(missing_val=apply(Cus_train,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val
#Finding the missing values in test data
missing_val<-data.frame(missing_val=apply(Cus_test,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val
##No missing values are present in both train and test data.
##Checking correlation between the attributes
#Correlations in train data
#convert factor to int
Cus_train$target<-as.numeric(Cus_train$target)
train_correlations<-cor(Cus_train[,c(2:202)])
train_correlations
#We can observed that the correlation between the train attributes is very small
#Correlations in test data
test_correlations<-cor(Cus_test[,c(2:201)])
test_correlations
##We can observed that the correlation between the test attributes is very small.
###Feature Engineering###
#Features on Train data
Cus_train$sum<-rowSums(Cus_train[,names(Cus_train)[c(2:202)]])
Cus_train$min<-apply(Cus_train[,-c(1,2)],MARGIN=1,FUN=min)
Cus_train$max<-apply(Cus_train[,-c(1,2,203,204,205,206)],MARGIN=1,FUN=max)
Cus_train$mean<-apply(Cus_train[,-c(1,2,203,204,205,206)],MARGIN=1,FUN=mean)
Cus_train$sd<-apply(Cus_train[,-c(1,2,203,204,205,206)],MARGIN=1,FUN=sd)
Cus_train$skewness<-apply(Cus_train[,-c(1,2,203,204,205,206,207)],MARGIN=1,FUN=skewness)
Cus_train$kurtosis<-apply(Cus_train[,-c(1,2,203,204,205,206,207,208)],MARGIN=1,FUN=kurtosis)
Cus_train$med<-apply(Cus_train[,-c(1,2,203,204,205,206,207,208,209)],MARGIN=1,FUN=median)
#Features on test data
Cus_test$sum<-rowSums(Cus_test[,names(Cus_test)[c(2:201)]])
Cus_test$min<-apply(Cus_test[,-c(1,201)],MARGIN=1,FUN=min)
Cus_test$max<-apply(Cus_test[,-c(1,201,202,203,204,205,206)],MARGIN=1,FUN=max)
Cus_test$mean<-apply(Cus_test[,-c(1,202,203,204,205,206,207,208)],MARGIN=1,FUN=mean)
Cus_test$sd<-apply(Cus_test[,-c(1,202,203,204,205,206,207,208)],MARGIN=1,FUN=sd)
Cus_test$skewness<-apply(Cus_test[,-c(1,202,203,204,205,206,207,208)],MARGIN=1,FUN=skewness)
Cus_test$kurtosis<-apply(Cus_test[,-c(1,202,203,204,205,206,207,208)],MARGIN=1,FUN=kurtosis)
Cus_test$med<-apply(Cus_test[,-c(1,202,203,204,205,206,207,208)],MARGIN=1,FUN=median)
dim(Cus_test)
dim(Cus_train)
##Let us build simple model to find features which are more important.

#Split the training data using simple random sampling
train_index<-sample(1:nrow(Cus_train),0.75*nrow(Cus_train))
#train data
train_data<-Cus_train[train_index,]
#validation data
valid_data<-Cus_train[-train_index,]
#dimension of train and validation data
dim(train_data)
dim(valid_data)
#setting memory limit
memory.limit()
memory.limit(100000)
#Training the Random forest classifier
set.seed(2732)
#convert to int to factor
train_data$target<-as.factor(train_data$target)
#setting the mtry
mtry<-floor(sqrt(200))
#setting the tunegrid
tuneGrid<-expand.grid(.mtry=mtry)
#fitting the ranndom forest
rf<-randomForest(target~.,train_data[,-c(1)],mtry=mtry,ntree=10,importance=TRUE)

#Variable importance
VarImp<-importance(rf,type=2)
VarImp
#Take away:
# We can observed that the top important features are var_12, var_26, var_22,v var_174, var_190 and 
#so on based on Mean decrease gini.
####Handling of imbalanced data####

#Split the data using CreateDataPartition
set.seed(689)
#train.index<-createDataPartition(train_df$target,p=0.8,list=FALSE)
train.index<-sample(1:nrow(Cus_train),0.8*nrow(Cus_train))
#train data
train.data<-Cus_train[train.index,]
#validation data
valid.data<-Cus_train[-train.index,]
#dimension of train data
dim(train.data)
#dimension of validation data
dim(valid.data)
#target classes in train data
table(train.data$target)
#target classes in validation data
table(valid.data$target)

# #function for calculating the FNR,FPR,Accuracy
calc <- function(cm){
  TN = cm[1,1]
  FP = cm[1,2]
  FN = cm[2,1]
  TP = cm[2,2]
  # #calculations
  print(paste0('Accuracy :- ',((TN+TP)/(TN+TP+FN+FP))*100))
  print(paste0('FNR :- ',((FN)/(TP+FN))*100))
  print(paste0('FPR :- ',((FP)/(TN+FP))*100))
  print(paste0('FPR :- ',((FP)/(TN+FP))*100))
  print(paste0('precision :-  ',((TP)/(TP+FP))*100)) 
  print(paste0('recall//TPR :-  ',((TP)/(TP+FP))*100))
  print(paste0('Sensitivity :-  ',((TP)/(TP+FN))*100))
  print(paste0('Specificity :-  ',((TN)/(TN+FP))*100))
  plot(cm)
}
calc(cm_RF)

####Model Develop ment########
###########Logistic Regression model###
#Training and validation dataset
#Training dataset
X_t<-as.matrix(train.data[,-c(1,2)])
y_t<-as.matrix(train.data$target)
#validation dataset
X_v<-as.matrix(valid.data[,-c(1,2)])
y_v<-as.matrix(valid.data$target)
#test dataset
test<-as.matrix(Cus_test[,-c(1)])
##########Logistic regression model###########
set.seed(667) # to reproduce results
lr_model <-glmnet(X_t,y_t, family = "binomial")
summary(lr_model)
#Cross validation prediction
set.seed(8909)
cv_lr <- cv.glmnet(X_t,y_t,family = "binomial", type.measure = "class")
cv_lr
#Plotting the missclassification error vs log(lambda) where lambda is regularization parameter
#Minimum lambda
cv_lr$lambda.min
#plot the auc score vs log(lambda)
plot(cv_lr)
#We can observed that miss classification error increases as increasing the log(Lambda).
#Model performance on validation dataset
set.seed(5363)
cv_predict.lr<-predict(cv_lr,X_v,s = "lambda.min", type = "class")
cv_predict.lr

#Confusion matrix
set.seed(689)
#actual target variable
target<-valid.data$target
#convert to factor
target<-as.factor(target)

#predicted target variable
#convert to factor
cv_predict.lr<-as.factor(cv_predict.lr)
confusionMatrix(data=cv_predict.lr,reference=target)
##Reciever operating characteristics(ROC)-Area under curve(AUC) score and curve
#ROC_AUC score and curve
set.seed(892)
cv_predict.lr<-as.numeric(cv_predict.lr)
plot(roc(valid.data$target,cv_predict.lr))

#Random Oversampling Examples(ROSE)
set.seed(699)
train.rose <- ROSE(target~., data =train.data[,-c(1)],seed=32)$data
#target classes in balanced train data
table(train.rose$target)
valid.rose <- ROSE(target~., data =valid.data[,-c(1)],seed=42)$data
#target classes in balanced valid data
table(valid.rose$target)
#Let us see how baseline logistic regression model performs on synthetic data points
#Logistic regression model
set.seed(462)
lr_rose <-glmnet(as.matrix(train.rose),as.matrix(train.rose$target), family = "binomial")
summary(lr_rose)
#Cross validation prediction
set.seed(473)
cv_rose = cv.glmnet(as.matrix(valid.rose),as.matrix(valid.rose$target),family = "binomial", type.measure = "class")
cv_rose
#Minimum lambda
cv_rose$lambda.min
#plot the auc score vs log(lambda)
plot(cv_rose)
#Model performance on validation dataset
set.seed(442)
cv_predict.rose<-predict(cv_rose,as.matrix(valid.rose),s = "lambda.min", type = "class")
cv_predict.rose
#Confusion matrix
set.seed(478)
#actual target variable
target<-valid.rose$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cv_predict.rose<-as.factor(cv_predict.rose)
#Confusion matrix
confusionMatrix(data=cv_predict.rose,reference=target)
#ROC_AUC score and curve
set.seed(843)
#convert to numeric
cv_predict.rose<-as.numeric(cv_predict.rose)
##plotting ROC Curve
plot(roc(valid.rose$target,cv_predict.rose))
##########KNN#######
### ##----------------------- KNN ----------------------- ## ###
set.seed(101)
## KNN impletation
library(FNN)

##Predicting Test data
#knn_Pred = knn(train = trainset[,1:14],test = validation_set[,1:14],cl = trainset$Churn, k = 5)

knn_Pred = knn(train = train.rose,test = valid.rose,cl = train.rose$target, k = 3,prob = T)
#Confusion matrix
cm_knn = table(valid.rose$target,knn_Pred)
confusionMatrix(cm_knn)
calc(cm_knn)
##plotting ROC Curve
knn_pred=as.numeric(knn_Pred)
plot(roc(valid.rose$target,knn_pred))
auc(knn_pred,valid.rose$target)
################# ##----------------------- Random Forest ----------------------- ## ###
 library(randomForest)

set.seed(101)
train.rose$target <- as.character(train.rose$target)
train.rose$target <- as.factor(train.rose$target)
RF_model = randomForest(target ~ ., train.rose,ntree= 100,importance=T,type='class')
plot(RF_model)
#Predict test data using random forest model
RF_Predictions = predict(RF_model, valid.rose[,-1])

##Evaluate the performance of classification model
cm_RF = table(valid.rose$target,RF_Predictions)
confusionMatrix(data=cm_RF,reference=target)
plot(RF_model)
calc(cm_RF)
##########################3
##plotting ROC Curve
rf_prediction=as.numeric(RF_Predictions)
plot(roc(valid.rose$target,rf_prediction))
auc(RF_Predictions,valid.rose$target)

####### ##----------------------- Naive Bayes ----------------------- ## ###

# library(e1071) #lib for Naive bayes
set.seed(101)
#Model Development and Training
naive_model = naiveBayes(target ~., data = train.rose, type = 'class')
#prediction
naive_pred = predict(naive_model,valid.rose[,-1])

#Confusion matrix
cm_naive = table(valid.rose$target,naive_pred)
confusionMatrix(cm_naive)
##plotting ROC
navie_pred=as.numeric(naive_pred)
plot(roc(valid.rose$target,navie_pred))
calc(cm_naive)
auc(naive_pred,valid.rose$target)
