# https://www.kaggle.com/kevinbonnes/basic-tf-idf-model-0-647-lb
# https://www.kaggle.com/headsortails/personalised-medicine-eda-with-tidy-r

library(readr)
library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(tm)
# Interesting package
# Extracts Sentiment and Sentiment-Derived Plot Arcs from Text
library(syuzhet)
library(caret)
library(Matrix)
library(xgboost)

setwd('git\\Kaggle\\Personalized-Medicine_Redefining-Cancer-Treatment')

# LabelCount Encoding function
labelCountEncoding = function(column){
  return(match(column,levels(column)[order(summary(column,maxsum=nlevels(column)))]))
}

train_text = as.data.table(read_lines('input/training_text', skip = 1))
train_text = train_text %>%
  separate(V1, into = c("ID", "Text"), sep = "\\|\\|")
train_text = train_text %>%
  mutate(ID = as.integer(ID))

test_text = as.data.table(read_lines('input/stage2_test_text.csv', skip = 1))
test_text = test_text %>%
  separate(V1, into = c("ID", "Text"), sep = "\\|\\|")
test_text = test_text %>%
  mutate(ID = as.integer(ID))

train = fread("input/training_variants", sep=",", stringsAsFactors = T)
test  = fread("input/stage2_test_variants.csv", sep=",", stringsAsFactors = T)
train = merge(train, train_text, by="ID")
test  = merge(test, test_text, by="ID")
rm(test_text, train_text)
gc()

# A way to mark the data which don't have to create a new col.
test$Class = -1
data = rbind(train, test)
rm(train, test)
gc()

# Basic text features
data$nchar  = nchar(data$Text)
data$nwords = str_count(data$Text, "\\S+") # \\S means not space.

# TF-IDF
txt = Corpus(VectorSource(data$Text))
txt = tm_map(txt, stripWhitespace)
txt = tm_map(txt, content_transformer(stringi::stri_trans_tolower))
txt = tm_map(txt, removePunctuation)
txt = tm_map(txt, removeWords, stopwords("english"))
txt = tm_map(txt, stemDocument, language="english")
# What's stem? https://www.r-bloggers.com/r-stem-pre-processed-text-blocks/
txt = tm_map(txt, removeNumbers)
dtm = DocumentTermMatrix(txt, control = list(weighting = weightTfIdf)) #Weight a term-document matrix by tf-idf.
dtm = removeSparseTerms(dtm, 0.95)
dtm = as.matrix(dtm)
# Not a good but fast way to pass the encoding error.
colnames(dtm) = iconv(colnames(dtm), 'latin1', 'latin1')
data = cbind(data, dtm)
rm(dtm)

# LabelCount Encoding for Gene and Variation
data$Gene = labelCountEncoding(data$Gene)
data$Variation = labelCountEncoding(data$Variation)

# Sentiment analysis
# To solve encoding error.
data$Text = iconv(data$Text, 'latin1', 'latin1')
sentiment = get_nrc_sentiment(data$Text) 
head(sentiment)
data = cbind(data,sentiment)

# Set seed
set.seed(1)
cvFoldsList = createFolds(data$Class[data$Class > -1], k=5)

# To sparse matrix
varnames = setdiff(colnames(data), c("ID", "Class", "Text"))
train_sparse = Matrix(as.matrix(sapply(data[Class > -1, varnames, with=FALSE], as.numeric)), sparse = TRUE)
test_sparse  = Matrix(as.matrix(sapply(data[Class == -1, varnames, with=FALSE], as.numeric)), sparse = TRUE)

#' "multi:softprob"
#' set xgboost to do multiclass classification using the softmax objective. 
#' Class is represented by a number and should be from 0 to num_class - 1.
y_train  = data[Class > -1, Class] - 1

test_ids = data[Class == -1, ID]
dtrain   = xgb.DMatrix(data=train_sparse, label=y_train)
dtest    = xgb.DMatrix(data=test_sparse)

# Params for xgboost
# https://rdrr.io/cran/xgboost/man/xgb.train.html
param = list(booster = "gbtree",
              objective = "multi:softprob", 
              eval_metric = "mlogloss",
              num_class = 9,
              eta = 0.2, #control the learning rate
              gamma = 1,
              max_depth = 5,
              #min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7
)

# Cross validation - determine CV scores & optimal amount of rounds
xgb_cv = xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 1000,
                 maximize = FALSE,
                 prediction = TRUE,
                 folds = cvFoldsList,
                 print.every.n = 5,
                 early.stop.round = 100
)
(rounds = xgb_cv$best_iteration)#which.min(xgb_cv$dt[, test.mlogloss.mean]))

# Train model
xgb_model = xgb.train(data = dtrain,
                       params = param,
                       watchlist = list(train = dtrain), # To watch the performance of each round's model on dtrain.
                       nrounds = rounds,
                       verbose = 1,
                       print_every_n = 5
)

# Feature importance
#names = dimnames(train_sparse)[[2]]
#importance_matrix = xgb.importance(names,model = xgb_model)
#xgb.plot.importance(importance_matrix[1:30, ], 10)

preds = as.data.table(t(matrix(predict(xgb_model, dtest), nrow=9, ncol=nrow(dtest))))
colnames(preds) = c("class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8", "class9")

write.csv(data.table(ID=test_ids, preds), "output/xgb_cv_submission.csv",row.names=F)