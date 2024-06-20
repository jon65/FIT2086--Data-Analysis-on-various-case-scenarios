source("my.prediction.stats.R")
source("wrappers.R")
library(glmnet)
library(rpart)
library(randomForest)
library(kknn)

#Q2.1
heart.train = read.csv("heart.train.2023.csv",stringsAsFactors = T)
heart.test = read.csv("heart.test.2023.csv",stringsAsFactors = T)
#fit a decision tree
tree.heart = rpart(HD~., heart.train)
#decision tree
tree.heart

#use cross validation with 10 folds and 5000 repetitions
cv = learn.tree.cv(HD~.,data=heart.train,nfolds=10,m=5000)
#optimal tree
cv$best.tree
#cross validation shows 7 is the best tree size
plot.tree.cv(cv)
#optimal number of leaf is 7

#Q2.2
#plot
plot(cv$best.tree)
text(cv$best.tree, pretty=12)

#Q2.5
#fit a logistic regression model to the data
fullmod=glm(HD ~ ., data=heart.train, family=binomial)
summary(fullmod)

#prune back model using BIC and both direction
#usng backwise selection and bic
fit_bic = step(fullmod, trace=0, k = log(nrow(heart.train)), direction="both")
summary(fit_bic)
#my.pred.stats(predict(fit_bic,heart.test,type="response"),heart.test$HD)
fit_bic$coefficients

#how do they compare with the variable used by the tree estimated by CV?
#TREE cv
summary(cv$best.tree)
cv$best.tree$variable.importance

#bic fit
summary(fit_bic)
fit_bic$coefficients


#Q2.6
summary(fit_bic)
fit_bic$coefficients

#Q2.7
#logistic BIC
my.pred.stats(predict(fit_bic,heart.test,type="response"),heart.test$HD)
# tree
my.pred.stats(predict(cv$best.tree, heart.test)[,2], heart.test$HD)

#Which predictor is the most important in the logistic regression?

#Q2.8
#2ways of finding
#logistic regression 
test = heart.test[69,]
bic_pred = predict(fit_bic, test, type="response")
#prob of HD
bic_pred
#prob of not HD
1-bic_pred
#odds - 17.63966
bic_pred/(1-bic_pred)

#odds - 17.63966
bic_pred = predict(fit_bic, test)
odds = exp(bic_pred)

#decision tree
predict(cv$best.tree, test)
#log odds = 6.333335
0.8636364/(1-0.8636364)

# 2.9
# for 69th patient
boot.odd = function(formula, data, indices)
{
  #Create a bootstrapped version of our data
  d = data[indices,]
  #get model
  pred = glm(formula, d, family=binomial)
  #get pred
  patient_69 = predict(pred, heart.test[69,], type="response")
  return(patient_69)
}

bs = boot(data=heart.train, statistic=boot.odd, R=5000, formula= HD ~ .)
plot(bs)



