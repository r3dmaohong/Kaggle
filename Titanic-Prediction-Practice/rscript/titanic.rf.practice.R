####################################
##Kaggle Tatanic prediction practice.
####################################

gc()
setwd(file.path('git', 'kaggle','Titanic-Prediction-Practice'))

dir.create('output', showWarnings = FALSE)
dir.create('rscript', showWarnings = FALSE)
dir.create('tmp', showWarnings = FALSE)

##Libraries
library(randomForest)
library(dplyr)
library(ggplot2)
library(ggthemes)

##
train <- read.csv("input/train.csv", na.strings=c('NA',''))
test  <- read.csv("input/test.csv", na.strings=c('NA',''))

train$memo <- 'train'
test$memo <- 'test'
test$Survived <- NA

total <- rbind(train, test)

total$FamilySize <- (total$Parch + total$SibSp + 1)
total$Title <- sapply(total$Name, function(x) substr(x, gregexpr(', ',x)[[1]][1]+2, gregexpr('. ', x, fixed=T)[[1]][1]-1))
unique(total$Title)
total$Title[total$Title=="Mlle" | total$Title=="Ms"] <- "Miss"
total$Title[total$Title=="Mme" | total$Title=="the Countess"] <- "Mrs"
total$Mother <- 0
total$Mother[total$Sex=='female' & total$Parch>0 & total$Age>18 & total$Title!='Miss'] <- 1
total$FamilySize[total$Title=="Dona"] %>% unique
total$FamilySize[total$Title=="Lady"] %>% unique
total$Title[total$Title=="Dona"] <- "Miss"

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
total$Title[total$Title %in% rare_title]  <- 'Rare Title'
total$Title <- factor(total$Title)

#Adding Child
total$Child <- 0
total$Child[total$Parch>0 & total$Age<=18] <- 1
#https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic/run/198371
# Use ggplot2 to visualize the relationship between family size & survival
ggplot(total[!is.na(total$Survived),], aes(x = FamilySize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()
# There’s a survival penalty to singletons and those with family sizes above 4

# Discretize family size
total$FamilySizeD[total$FamilySize == 1] <- 'singleton'
total$FamilySizeD[total$FamilySize <= 4 & total$FamilySize > 1] <- 'small'
total$FamilySizeD[total$FamilySize > 4] <- 'large'
total$FamilySizeD <- factor(total$FamilySizeD)

# Show family size by survival using a mosaic plot
mosaicplot(table(total$FamilySizeD, total$Survived), main='Family Size by Survival', shade=TRUE)

variable.names <- c('Cabin', 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','Mother','Child','FamilySizeD') 
##Find out which col has NA values.
tmp <- apply(total, 2, function(x) sum(is.na(x), na.rm=T) ) %>% data.frame
tmp <- rownames(tmp)[which(tmp[,1]>0)]
##col "Age" has NA values.

#Remove rows with NA value.
#train[complete.cases(train),] %>% nrow
#train <- train[complete.cases(train),]

## Predict NA values.
##Maybe I can use for loop and eval to replace NA values automatically
tmp
tmp <- setdiff(tmp, "Survived")
cat("TRUE : ", tmp[tmp %in% variable.names])
cat("FALSE : ", tmp[!tmp %in% variable.names])

total$Cabin %>% summary()
total$ttl_Cabin <- total$Cabin
total$Cabin <- substr(gsub("[0-9]", "", total$Cabin), 1, 1)
total$Cabin <- as.factor(total$Cabin)

which(is.na(total$Embarked)) 
# Get rid of our missing passenger IDs
embark_fare <- total %>%
  filter(PassengerId != 62 & PassengerId != 830)

cat(paste('We will infer their values for **embarkment** based on present data that we can imagine may be relevant: **passenger class** and **fare**. We see that they paid<b> $', total[c(62, 830), 'Fare'][[1]][1], 'and $', total[c(62, 830), 'Fare'][[1]][2], 'respectively and their classes are', total[c(62, 830), 'Pclass'][[1]][1], 'and', total[c(62, 830), 'Pclass'][[1]][2], '. So from where did they embark?'))

# Use ggplot2 to visualize embarkment, passenger class, & median fare
library(scales)
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

# Since their fare was $80 for 1st class, they most likely embarked from 'C'
#total$Embarked[c(62, 830)] <- 'C'

i=which(tmp=="Embarked")
rf.call <- paste0("randomForest(",tmp[i],"~",setdiff(variable.names,tmp) %>% paste0(collapse="+"),", data=total[complete.cases(total[,c(setdiff(variable.names,tmp),tmp[i])]),c(setdiff(variable.names,tmp),tmp[i])], importance=TRUE, proximity=TRUE, ntree=2000)")
tmp.rf <- eval(parse(text=rf.call))
(tmp.prediction <- predict(tmp.rf, total[is.na(total[,tmp[i]]),]))
## Same result: C
total[is.na(total[,tmp[i]]),tmp[i]] <- tmp.prediction

# Show row 1044
total[1044, ]
ggplot(total[total$Pclass == '3' & total$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()
## From this visualization, 
##it seems quite reasonable to replace the NA Fare value with median 
##for their class and embarkment which is $8.05.
median(total[total$Pclass == '3' & total$Embarked == 'S', ]$Fare, na.rm = TRUE)

which(is.na(total$Fare))
tmp <- setdiff(tmp, "Embarked")
i=which(tmp=="Fare")
rf.call <- paste0("randomForest(",tmp[i],"~",setdiff(variable.names,tmp) %>% paste0(collapse="+"),", data=total[complete.cases(total[,c(setdiff(variable.names,tmp),tmp[i])]),c(setdiff(variable.names,tmp),tmp[i])], importance=TRUE, proximity=TRUE, ntree=2000)")
tmp.rf <- eval(parse(text=rf.call))
(tmp.prediction <- predict(tmp.rf, total[is.na(total[,tmp[i]]),]))
##  hmm... $11.68 VS $8.05
ggplot(total[total$Pclass == '3' & total$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  geom_vline(aes(xintercept=tmp.prediction),
             colour='blue', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()
## Seems that using median is a better solution...?
## Still use rf prediction result...
##total[is.na(total[,tmp[i]]),tmp[i]] <- tmp.prediction
total[is.na(total[,tmp[i]]),tmp[i]] <- median(total[total$Pclass == '3' & total$Embarked == 'S', ]$Fare, na.rm = TRUE)


##
# Finally, grab surname from passenger name
sapply(total$Name, 
       function(x) strsplit(as.character(x), split = '[,.]')[[1]][1]) %>% table

# Set a random seed
set.seed(129)
library(mice)
# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(total[, !names(total) %in% c('PassengerId','Name','Ticket','Cabin','Survived', 'ttl_Cabin')], method='rf') 
# Save the complete output 
mice_output <- complete(mice_mod)

which(is.na(total$Age))
tmp <- setdiff(tmp, "Fare")
i=which(tmp=="Age")
rf.call <- paste0("randomForest(",tmp[i],"~",setdiff(variable.names,tmp) %>% paste0(collapse="+"),", data=total[complete.cases(total[,c(setdiff(variable.names,tmp),tmp[i])]),c(setdiff(variable.names,tmp),tmp[i])], importance=TRUE, proximity=TRUE, ntree=2000)")
tmp.rf <- eval(parse(text=rf.call))
(tmp.prediction <- predict(tmp.rf, total[is.na(total[,tmp[i]]),]))

#knn
library(DMwR)
imputeData <- knnImputation(total[,which(sapply(total, class)=="integer" | sapply(total, class)=="numeric")])
imputeData$Age

# Plot age distributions
par(mfrow=c(1,4))
hist(total$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))
hist(predict(tmp.rf, total), freq=F, main='Age: rf Output', 
     col='blue', ylim=c(0,0.04))
hist(imputeData$Age, freq=F, main='Age: knn Output', 
     col='red', ylim=c(0,0.04))
par(mfrow=c(1,1))
##hmm.. MICE is much better.
# Replace Age variable from the mice model.
total$Age <- mice_output$Age
tmp <- setdiff(tmp, "Age")

# First we'll look at the relationship between age & survival
ggplot(total[!is.na(total$Survived),], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # I include Sex since we know (a priori) it's a significant predictor
  facet_grid(.~Sex) + 
  theme_few()

table(total$Child, total$Survived)
table(total$Mother, total$Survived)

#Display missing-data patterns.
md.pattern(total)

#' Cabin feature may be dropped as it is highly incomplete or 
#' contains many null values both in training and test dataset.


if(toString(tmp)!=""){
  for(i in 1:length(tmp)){
    if(tmp[i] %in% variable.names){
      rf.call <- paste0("randomForest(",tmp[i],"~",setdiff(variable.names,tmp) %>% paste0(collapse="+"),", data=total[complete.cases(total[,c(setdiff(variable.names,tmp),tmp[i])]),c(setdiff(variable.names,tmp),tmp[i])], importance=TRUE, proximity=TRUE, ntree=2000)")
      tmp.rf <- eval(parse(text=rf.call))
      tmp.prediction <- predict(tmp.rf, total[is.na(total[,tmp[i]]),])
      total[is.na(total[,tmp[i]]),tmp[i]] <- tmp.prediction
    }  
  }
}
# hmm.. cause of too many na, maybe the prediction result isn't that correct... 
total$Cabin %>% table

train <- total[total$memo=="train",]
test  <- total[total$memo=="test",]



prediction.tv <-"Survived"
rf.call <- paste0("randomForest(as.factor(",prediction.tv,")~",variable.names %>% paste0(collapse="+"),", data=train, importance=TRUE, proximity=TRUE, ntree=2000)")
titanic.rf <- eval(parse(text=rf.call))
varImpPlot(titanic.rf)

# Show model error
plot(titanic.rf, ylim=c(0,0.36))
legend('topright', colnames(titanic.rf$err.rate), col=1:3, fill=1:3)

## remove useless col to get higher accuracy of prediction
rf.call <- paste0("randomForest(as.factor(",prediction.tv,")~",setdiff(variable.names, c("Cabin", "FamilySize", "Mother", "Child","Parch","Embarked")) %>% paste0(collapse="+"),", data=train, importance=TRUE, proximity=TRUE, ntree=2000, do.trace=T)")
titanic.rf <- eval(parse(text=rf.call))
varImpPlot(titanic.rf)


# Show model error
plot(titanic.rf, ylim=c(0,0.36))
legend('topright', colnames(titanic.rf$err.rate), col=1:3, fill=1:3)

## Much lower without Cabin

test$Survived <- predict(titanic.rf, test)

prediction.output <- test[,c("PassengerId","Survived")]

write.csv(prediction.output,'output\\test_prediction.csv',row.names=F)


####
titanic.rf$confusion

importance    <- importance(titanic.rf)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))
(rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance)))))

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()
