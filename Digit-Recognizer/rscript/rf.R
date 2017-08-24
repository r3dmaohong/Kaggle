http://rischanlab.github.io/RandomForest.html

rm(list = ls())
gc()

original_path <- getwd()

##Libraries
library(randomForest)
library(data.table)

setwd(file.path("Kaggle", "Digit-Recognizer"))
dir.create('output', showWarnings = FALSE)

nTrees <- 2000

train_data <- fread("input/train.csv")
test <- fread("input/test.csv")

head(names(train_data), 10)
dim(train_data)

max(train_data)
#train_data[,-1] = train_data[,-1]/255
train_data = train_data/255
train_data$label = train_data$label*255
table(train_data$label)
max(test)
test = test/255

##
smp_size <- floor(0.7 * nrow(train_data))
set.seed(1)
train_ind <- sample(seq_len(nrow(train_data)), size = smp_size)

train <- train_data[train_ind, ]
train_valid <- train_data[-train_ind, ]

rf <- randomForest(train[,-1], as.factor(train[[1]]), proximity=FALSE, ntree=nTrees)
table(predict(rf), train[[1]])
print(rf)
plot(rf)
#importance(rf)
#varImpPlot(rf)

output <- data.frame(ImageId=1:nrow(test), Label=predict(rf, test))

write.csv(output, "output/r_rf_test.csv", row.names=F)
