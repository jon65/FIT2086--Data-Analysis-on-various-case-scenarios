#Question 1
rm(list=ls())
housing = read.csv("housing.2023.csv")

#Q1.1
fit = lm(medv ~., data = housing)
summary(fit)

#Q1.2
#using the bonferroni we must divide the alpha by the number of tests performed p 
pvalues = coefficients(summary(fit))[,4]
#12 predictors
p = 12
pvalues < 0.05/p
sum(pvalues < 0.05/p)
#6 predictors passed the Bonferroni procedure
pvalues < 0.05
sum(pvalues < 0.05)
#This is 4 less than 0.05 significance tests 
#dropped crim, nox, rad and tax after bonferroni
# The Bonferroni gives us confidence that anything that does pass the threshold is likely
# a real association at the population level, 
#but it is very conservative and will often discover
# nothing as the evidence required to overcome the threshold is too high.

#Q1.3
#Describe what effect the per-capita crime rate (crim) appears to have on the median house price.

#for every unit increase in crime rate, there the median house price will decrease by 0.115817989.

#Q1.4
n = length(housing$medv)
fit_bic = step(fit, direction="both", k = log(n))
summary(fit_bic)
fit_bic$coefficients
# E[medv] = 29.1926650 + 4.5991149*chas -17.3765139*nox + 4.8206454*rm  -0.9359373*dis  -0.9591382*ptratio  -0.4947192* lstat

#
#Q1.6
# Create a data frame with the predictor values for the new suburb
new_suburb_data <- data.frame(
  crim = 0.04741,
  zn = 0,
  indus = 11.93,
  chas = 0,
  nox = 0.573,
  rm = 6.03,
  age = 80.8,
  dis = 2.505,
  rad = 1,
  tax = 273,
  ptratio = 21,
  lstat = 7.88
)

pred = predict(fit_bic, new_suburb_data)
pred
pred_ci = predict(fit_bic, new_suburb_data, interval="confidence")
pred_ci
#fit      lwr      upr
#1 21.9196 20.30209 23.53712

#Q1.7
#interaction between number of rooms a dwelling has 
fit_interact <- lm(medv ~ rm * dis, data = housing)
summary(fit_interact)
summary(fit_bic)
# dis by its own is not associated with median house value but rm is heavily associated 
#p-value of interactions between number of rooms a dwelling has and its distance to one of the employment centres is 0.065
# which suggest some association 
#  residual standard error has gone up to 6.548 from 5.239 which suggest data is uncorrelated but could be due to small sample size.
# which potentially
#indicates positive effect of the interaction between rm and dis on median house prices although dis by itself has no association.

#Question 2
rm(list=ls())

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


#QUESTION 3
rm(list=ls())
#Q3.1

source("my.prediction.stats.R")
source("wrappers.R")
library(boot)
library(kknn)

ms_train = read.csv('ms.measured.2023.csv', header=T, stringsAsFactors=T)
ms_test = read.csv('ms.truth.2023.csv', header=T, stringsAsFactors=T)
x = c(1:25)
y = c()

# fit KKNN model
for (i in 1:25){
  
  prediction = fitted(kknn(intensity~ ., ms_train, ms_test, kernel='optimal', k=i))
  mse = mean((prediction - ms_test$intensity)^2)
  
  cat('k value:',i,' mean-squared error:',mse,'\n')
  y = c(y,mse)
  
}


# Produce a plot of plot of these errors against the various values of k
plot(x, y, xlab='k', ylab='Mean-squared error (MSE)', main='Mean squared error against k values used')

#Q3.2

# k = 2

#predict test values for k =2
ms_predict2 = fitted(kknn(intensity~ ., ms_train, ms_test, kernel='optimal', k=2))

#plot training data data 
plot(ms_train$MZ, ms_train$intensity, type='l', col='black', xlab='Mass to charge ratio (MZ)', ylab='Intensity of ion (Intensity)', 
     main='k=2')

#plot test data
lines(ms_test$MZ, ms_test$intensity, type='l', col='blue')

#plot predicted data
lines(ms_test$MZ, ms_predict2, type='l', col='red')

legend('topright', legend=c('train data','true spectrum','Estimated spectrum'), col=c('black','blue','red'), lwd=2, lty=c(1,1))


# k = 5
ms_predict5 = fitted(kknn(intensity~ ., ms_train, ms_test, kernel='optimal', k=5))

#plot training data data 
plot(ms_train$MZ, ms_train$intensity, type='l', col='black', xlab='Mass to charge ratio (MZ)', ylab='Intensity of ion (Intensity)', 
     main='k=5')

#plot test data
lines(ms_test$MZ, ms_test$intensity, type='l', col='blue')

#plot predicted data
lines(ms_test$MZ, ms_predict5, type='l', col='red')

legend('topright', legend=c('train data','true spectrum','Estimated spectrum'), col=c('black','blue','red'), lwd=2, lty=c(1,1))


# k = 10
ms_predict10 = fitted(kknn(intensity~ ., ms_train, ms_test, kernel='optimal', k=10))

#plot training data data 
plot(ms_train$MZ, ms_train$intensity, type='l', col='black', xlab='Mass to charge ratio (MZ)', ylab='Intensity of ion (Intensity)', 
     main='k=10')

#plot test data
lines(ms_test$MZ, ms_test$intensity, type='l', col='blue')

#plot predicted data
lines(ms_test$MZ, ms_predict10, type='l', col='red')

legend('topright', legend=c('train data','true spectrum','Estimated spectrum'), col=c('black','blue','red'), lwd=2, lty=c(1,1))


# k = 25
ms_predict25 = fitted(kknn(intensity~ ., ms_train, ms_test, kernel='optimal', k=25))

#plot training data data 
plot(ms_train$MZ, ms_train$intensity, type='l', col='black', xlab='Mass to charge ratio (MZ)', ylab='Intensity of ion (Intensity)', 
     main='k=25')

#plot test data
lines(ms_test$MZ, ms_test$intensity, type='l', col='blue')

#plot predicted data
lines(ms_test$MZ, ms_predict25, type='l', col='red')

legend('topright', legend=c('train data','true spectrum','Estimated spectrum'), col=c('black','blue','red'), lwd=2, lty=c(1,1))

#3.4
#k=7
ms_predict25 = fitted(kknn(intensity~ ., ms_train, ms_test, kernel='optimal', k=7))

#plot training data data 
plot(ms_train$MZ, ms_train$intensity, type='l', col='black', xlab='Mass to charge ratio (MZ)', ylab='Intensity of ion (Intensity)', 
     main='k=7')

#plot test data
lines(ms_test$MZ, ms_test$intensity, type='l', col='blue')

#plot predicted data
lines(ms_test$MZ, ms_predict25, type='l', col='red')

legend('topright', legend=c('train data','true spectrum','Estimated spectrum'), col=c('black','blue','red'), lwd=2, lty=c(1,1))

#fit linear model
f = my.make.formula("intensity", ms_train, use.interactions=T, use.logs=T, use.squares=T, use.cubics=T)
fit = glm(f, data=ms_train)
fit_bic = step(fit, k = log(length(ms_train$intensity)), direction="both")
yhat_full_bic = predict(fit_bic, ms_test)
lines(ms_test$MZ, yhat_full_bic, type='l', col='brown')
legend('topright', legend=c('train data','true spectrum','Estimated spectrum', 'logistic models estimate'), col=c('black','blue','red', 'brown'), lwd=2, lty=c(1,1))

#Q3.5
#train knn
knn = train.kknn(intensity ~ ., data = ms_train, kmax=25, kernel="optimal")

#best value for k = 6
knn$best.parameters
#
bestk_ms = fitted( kknn(intensity ~ ., ms_train, ms_test,
                        kernel = knn$best.parameters$kernel, k = knn$best.parameters$k) )

#Q3.6
#cal residuals
residuals = ms_test$intensity - bestk_ms

#mean of residuals
mean_residuals = mean(residuals)

# Sum of squared error SSE
sum_squared_residuals = sum((residuals - mean_residuals)^2)

# Divide by (N-1) where N is the number of data points in the test set
n = length(ms_test$intensity)
var = sum_squared_residuals / (n - 1) # using unbiased

# Take the square root to get the standard deviation
sd_noise = sqrt(var)
sd_noise

#Q3.7
# find the index in which max peaks occurs
idx = which.max(bestk_ms)
idx

# find the MZ for the index found 
mz = ms_test$MZ[idx]
mz
#MZ intensity
#283 7963.3    96.638

#Q3.8
boot.intensity = function(data, indices, k_value)
{
  # Create a bootstrapped version of our data
  d = data[indices,]
  prediction = fitted(kknn(intensity ~ ., d, ms_test, kernel = "optimal", k = k_value) )  
  return(prediction[idx])
}

# k=3
#fit model
#find CI using Bootstraping
ms_test[idx,]
bs_intensity = boot(data=ms_train, statistic=boot.intensity, R=5000, k_value=3)
boot.ci(bs_intensity,conf=0.95,type="bca")

# k=6 - optimal found
ms_test[idx,]
bs_intensity = boot(data=ms_train, statistic=boot.intensity, R=5000, k_value=6)
boot.ci(bs_intensity,conf=0.95,type="bca")

# k=20
ms_test[idx,]
bs_intensity = boot(data=ms_train, statistic=boot.intensity, R=5000, k_value=20)
boot.ci(bs_intensity,conf=0.95,type="bca")






















