# Creates a simple random forest benchmark
#https://www.kaggle.com/benhamner/digit-recognizer/random-forest-benchmark/code
rm(list = ls()) #Remove all objects in the environment
gc() ##Free up the memory

original_path <- getwd()

##Libraries
library(randomForest)
library(readr)

setwd(file.path("Kaggle", "Digit-Recognizer"))
dir.create('output', showWarnings = FALSE)


numTrain <- nrow(train) #10000
numTrees <- 2000

train <- read_csv("input/train.csv")
test <- read_csv("input/test.csv")

head(names(train), 10)
nrow(train)
length(unique(train[,1])) #1~9

##cannot allocate vector of size X.X Gb
#rf <- randomForest(as.factor(label)~.,data = train, importance=TRUE, proximity=TRUE, ntree=numTrain) 
#set.seed(0)
rows <- sample(1:nrow(train), numTrain)
labels <- as.factor(train[rows,1])
head(labels)
train <- train[rows,-1]

rf <- randomForest(train, labels, xtest=test, ntree=numTrees) #a data frame or matrix (like x) containing predictors for the test set.
predictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])

write_csv(predictions, "output/rf_benchmark2.csv") #0.96757
##write.csv(predictions, "output/rf_benchmark.csv", row.names=F)
