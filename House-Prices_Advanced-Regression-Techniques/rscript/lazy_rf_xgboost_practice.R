#' https://www.kaggle.com/skirmer/fun-with-real-estate-data

setwd("git/Kaggle/House-Prices_Advanced-Regression-Techniques")

library(corrplot)
library(magrittr)

train = read.csv("input/train.csv", stringsAsFactors = FALSE)
test  = read.csv("input/test.csv", stringsAsFactors = FALSE)

dim(train)
head(train)

sapply(train, class) %>% table
num_ind = which(sapply(train, class)=="integer")

correlations = cor(train[,num_ind])
corrplot(correlations)

k = arrayInd(which(correlations>0.5 & correlations!=1), 
              dim(correlations))
apply(k,1, function(x) dimnames(correlations))

high_corr = cbind(sapply(1:nrow(k),  function(x) dimnames(correlations)[[1]][k[x,1]]),
      sapply(1:nrow(k),  function(x) dimnames(correlations)[[1]][k[x,2]]),
      correlations[k])

high_corr[,1:2] = t(apply(high_corr[,1:2], 1, sort))
( high_corr = unique( high_corr[order(high_corr[,1]),] ) )

pairs(~YearBuilt+OverallQual+TotalBsmtSF+GrLivArea,data=train,
      main="Simple Scatterplot Matrix")

library(car)
scatterplot(SalePrice ~ YearBuilt, data=train,  xlab="Year Built", ylab="Sale Price", grid=FALSE)
scatterplot(SalePrice ~ YrSold, data=train,  xlab="Year Sold", ylab="Sale Price", grid=FALSE)

test$SalePrice = 0
test$memo = "test"
train$memo = "train"
total = rbind(train, test)

# cols with na values
nacols = unique( arrayInd(which(is.na(total)), dim(total))[,2] )
names(total)[ nacols ]

head( total[, nacols ] , 25)
#which(sapply(nacols, 
#             function(x) class(total[, x])) == 
#        "integer")
#names(total)[.Last.value]

#' LotFrontage
#' MasVnrArea
#' GarageYrBlt
library(randomForest)

nrow(total)
sum(is.na(total$LotFrontage))

ind2factor = which(sapply(1:ncol(total), function(x) class(total[[x]]))=="character")

#total[, ind2factor] = apply(total[,ind2factor], 2, as.factor)
for(i in ind2factor){
  total[,i] = as.factor(total[,i])
}

for(i in 1:length(nacols)){
  rf4na  = paste0("randomForest(", names(total)[nacols][i], "~., data = total[!is.na(total$", names(total)[nacols][i], "),-c(1, setdiff( nacols, nacols[i] ) )], importance=TRUE, proximity=TRUE, ntree=2000)")
  tmp.rf = eval(parse(text=rf4na))
  total[is.na(total[,nacols[i]]), nacols[i]] = predict(tmp.rf, total[is.na(total[,nacols[i]]),-c(1, setdiff( nacols, nacols[i] ) )])
  print(paste0(i, ". ", names(total)[nacols][i], " completed...."))
}

nrow(test)
test_nacols = unique( arrayInd(which(is.na(test)), dim(test))[,2] )
nrow(train)
train_nacols = unique( arrayInd(which(is.na(train)), dim(train))[,2] )
## cols should be removed
total = total[,- union(
  test_nacols[sapply(unique( arrayInd(which(is.na(test)), dim(test))[,2] ), function(x) sum(is.na(test[,x]))) / nrow(test) > 0.3],
  train_nacols[sapply(unique( arrayInd(which(is.na(train)), dim(train))[,2] ), function(x) sum(is.na(train[,x]))) / nrow(train) > 0.3]
)]

train = total[total$memo=="train",]
test = total[total$memo=="test",]
train$memo = NULL
test$memo = NULL
test$SalePrice = NULL

library(caret)
partition = createDataPartition(y=train$SalePrice,
                                 p=.5,
                                 list=F)
training = train[partition,]
testing = train[-partition,]

################
#' random forest
################
rf_mdl = randomForest(SalePrice ~ ., data=training)


# Predict using the test set
prediction = predict(rf_mdl, testing)
model_output = cbind(testing, prediction)

model_output$log_prediction = log(model_output$prediction)
model_output$log_SalePrice = log(model_output$SalePrice)

library(ModelMetrics)
rmse(model_output$log_SalePrice,model_output$log_prediction)
rf_prediction = predict(rf_mdl, test)

###
###
##

rf_mdl = randomForest(SalePrice ~ ., data=train, proximity=FALSE, ntree=2000)
rf_prediction = predict(rf_mdl, test)

write.csv(cbind(Id=test$Id, SalePrice=rf_prediction),"output/rf_prediction.csv", row.names=F)
##########
#' xgboost
##########
#training$log_SalePrice = log(training$SalePrice)
#testing$log_SalePrice = log(testing$SalePrice)

library(Matrix)
xdata = sparse.model.matrix(SalePrice ~ .-1, data = training)

set.seed(1111)
library(xgboost)
xgb_mdl = xgboost(data = xdata,
                    label = training$SalePrice,
                    nrounds = 1000,
                    min_child_weight = 0,
                    max_depth = 10,
                    eta = 0.02,
                    subsample = .7,
                    colsample_bytree = .7,
                    booster = "gbtree",
                    eval_metric = "rmse",
                    verbose = TRUE,
                    print_every_n = 50,
                    nfold = 4,
                    nthread = 2,
                    objective="reg:linear")

Ypred = predict(xgb_mdl, sparse.model.matrix(SalePrice ~ .-1, data = testing))
rmse(log(testing$SalePrice), log(Ypred))
xgb_prediction = predict(xgb_mdl, sparse.model.matrix(~ ., data = test))




###
xdata = sparse.model.matrix(SalePrice ~ .-1, data = train)

set.seed(1111)
library(xgboost)
xgb_mdl = xgboost(data = xdata,
                  label = train$SalePrice,
                  nrounds = 600,
                  min_child_weight = 1,
                  max_depth = 10,
                  eta = 0.03,
                  subsample = .7,
                  colsample_bytree = 0.4,
                  booster = "gbtree",
                  eval_metric = "rmse",
                  verbose = TRUE,
                  print_every_n = 100,
                  nthread = 2,
                  objective="reg:linear")


xgb_prediction = predict(xgb_mdl, sparse.model.matrix(~ ., data = test))


write.csv(cbind(Id=test$Id, SalePrice=xgb_prediction),"output/xgb_prediction.csv", row.names=F)

#
#

#useless
#write.csv(cbind(Id=test$Id, SalePrice=mean(rowMeans(cbind(rf_prediction, xgb_prediction)))),"output/mean_prediction.csv", row.names=F)


#
# 
if(F){#library(devtools)
  #install_github("cran/neuralnet")
  library(neuralnet)
  
  factor_ind = which(sapply(1:ncol(train), function(x) class(train[,x]))=="factor")
  for(i in factor_ind){
    train = cbind(train, class.ind(train[,i]))
  }
  train = train[,-factor_ind]
  names(train) = gsub(".", "_", names(train), fixed=T)
  #f1 <- as.formula(paste0('SalePrice  ~ ', paste(names(train)[names(train)!="SalePrice"], collapse = "+")))
  f1 <- as.formula(paste("SalePrice~", paste(sprintf("`%s`", names(train)[names(train)!="SalePrice"]), collapse="+")))
  bpn <- neuralnet(formula = f1, data = train, hidden = c(1,1),learningrate = 0.01)
  
}