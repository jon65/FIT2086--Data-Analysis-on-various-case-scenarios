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










