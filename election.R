getwd()
setwd("C:/R/ABA")

# Import data:
electiondata <- read.csv("election_campaign_data.csv", sep=",", header=T, strip.white = T, na.strings = c("NA","NaN","","?")) 

# Explore data:
nrow(electiondata)
summary(electiondata)

#drop the variables
electiondata$cand_id <- NULL
electiondata$last_name <- NULL
electiondata$first_name <- NULL
electiondata$twitterbirth <- NULL
electiondata$facebookdate <- NULL
electiondata$facebookjan <- NULL
electiondata$youtubebirth <- NULL
#alternate way to drop columns
#mydata <- mydata[,c(1,2,5:7,9:15)]

#5.	Convert the following variables into factor variables 
electiondata$twitter <- as.factor(electiondata$twitter)
electiondata$facebook <- as.factor(electiondata$facebook)
electiondata$youtube <- as.factor(electiondata$youtube)
electiondata$cand_ici <- as.factor(electiondata$cand_ici)
electiondata$gen_election <- as.factor(electiondata$gen_election)
summary(electiondata)

# Install packages required for random forest:
#install.packages("randomForest")
library("randomForest")

# First, remove incomplete observations and store it in new dataframe 
electiondata1 <- electiondata[complete.cases(electiondata),]
summary(electiondata1)

## Create the training and test data:
n = nrow(electiondata1) # n will be ther number of obs. in data
trainIndex = sample(1:n, 
                    size = round(0.7*n), 
                    replace=FALSE) # We create an index for 70% of obs. by random

train_data = electiondata1[trainIndex,] # We use the index to create training data
test_data = electiondata1[-trainIndex,] # We take the remaining 30% as the testing data
summary(train_data)
summary(test_data)

set.seed(32) 
#build a random forest object
randfrst <- randomForest(train_data$gen_election ~ ., data=train_data, ntree=10, na.action=na.exclude, importance=T,
                  proximity=T) 
print(randfrst)

#For loop to test for different ntree values
for(i in seq(from=20, to=100,by=10))
{
  randfrstn <- randomForest(train_data$gen_election ~ ., data=train_data, ntree=i, na.action=na.exclude, importance=T,
                            proximity=T)
  print(randfrstn)
}

#to find optimum mtry, used the ntree with less OOB error rate 
mtry <- tuneRF(train_data[-26], train_data$gen_election, ntreeTry=50, 
               stepFactor=1.5, improve=0.01, trace=TRUE, plot=TRUE, na.action=na.exclude)

#find the mtry with lowest OOB
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

#use the mtry and ntree output by previous steps
set.seed(32)
randfrst <-randomForest(train_data$gen_election~., data=train_data, ntree=50, mtry=7, na.action=na.exclude, 
                        importance=T,proximity=T) 
print(randfrst)

# Calculate predictive probabilities of training dataset.
pred_train = predict(randfrst,type = "prob")
head(pred_train)

# Use the rf classifier to make the predictions(probs) on test set
pred_test_prob = predict(randfrst, type = "prob", test_data) 
# Add the predictions to test_data
final_pred_test <- cbind(test_data, pred_test_prob) 
# Add the new column names to the original column names 
colnames(final_pred_test) <- c(colnames(test_data),"prob.zero","prob.one") 
head(final_pred_test)
library(ROCR)
library(caret)
#creating a single column with W or L depending on predicted values
threshold <- 0.5
pred_test_factor <- factor(ifelse(pred_test_prob[,2] > threshold,'W','L'))
head(pred_test_factor)
levels(pred_test_factor)
levels(test_data$gen_election)
(confusionMatrix(pred_test_factor, test_data$gen_election, 
                positive = levels(test_data$gen_election)[2]))

#auc
(predict <- prediction(pred_test_prob[,2], test_data$gen_election))
(perf <- performance(predict, measure = "tpr", x.measure = "fpr"))
auc <- performance(predict, measure = "auc")
auc <- auc@y.values[[1]]
#roc curve
roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="Random Forest")

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))


#Evaluate variable importance
importance(randfrst)
#the feature on top of the plot is the important field
varImpPlot(randfrst)


# write the csv file of the output
#write.table(final_data, file="RF_predictions.csv", sep=",", row.names=F, col.names = colnames) 

# Install packages required for random forest:
install.packages("nnet")
# Load packages required for neural network
library(nnet)

# Size is the number of units (nodes) in the hidden layer.
ann <- nnet(train_data$gen_election~., data=train_data, size=5, maxit=1000) 
summary(ann)
print(ann)

# Use the ann classifier to make the predictions on test set
pred_test_ann = predict(ann, type = "raw", test_data) 
# Add the predictions to test_data
final_data_ann <- cbind(test_data, pred_test_ann) 
# Add the new column names to the original column names 
colnames <- c(colnames(test_data),"prob.one") 
head(pred_test_ann)
library(ROCR)
library(caret)
#creating a single column with 0 or 1 depending on predicted values
threshold <- 0.5
pred_test_ann_factor <- factor(ifelse(pred_test_ann[,1] > threshold,'W','L'))
head(pred_test_ann_factor)
(confusionMatrix(pred_test_ann_factor, test_data$gen_election, 
                 positive = levels(test_data$gen_election)[2]))

#auc
(predict <- prediction(pred_test_ann[,1], test_data$gen_election))
(perf <- performance(predict, measure = "tpr", x.measure = "fpr"))
auc <- performance(predict, measure = "auc")
auc <- auc@y.values[[1]]
#roc curve
roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="Neural Net-HL5")

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))

#increasing the hidden units to 20,30,23,24,25
#For loop to test for different ntree values
for(h in seq(from=6, to=30,by=1))
{
  ann1 <- nnet(train_data$gen_election~., data=train_data, size=h, maxit=1000) 
  #print(h)
  print(paste0("Hidden layer" , h , "converged"))
}
#Error in nnet.deafult , too many(1026) weights at no.of hidden units as 25, so using max value 24
ann1 <- nnet(train_data$gen_election~., data=train_data, size=24, maxit=1000) 
print(ann1)

# Use the ann classifier to make the predictions on test set
pred_test_ann1 = predict(ann, type = "raw", test_data) 
final_data_ann1 <- cbind(test_data, pred_test_ann1) 
colnames <- c(colnames(test_data),"prob.one") 
pred_test_ann_factor1 <- factor(ifelse(pred_test_ann1[,1] > threshold,'W','L'))
head(pred_test_ann_factor1)
(confusionMatrix(pred_test_ann_factor1, test_data$gen_election, 
                 positive = levels(test_data$gen_election)[2]))

#auc
(predict <- prediction(pred_test_ann1[,1], test_data$gen_election))
(perf <- performance(predict, measure = "tpr", x.measure = "fpr"))
auc <- performance(predict, measure = "auc")
auc <- auc@y.values[[1]]
auc
#roc curve
roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="Random Forest")

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))

#ftable for social media
attach(electiondata1)
ftable(xtabs(~facebook+twitter+youtube+gen_election, data=electiondata1)) 
