##Kaggle Tatanic prediction practice.
#rm(list = ls()) #Remove all objects in this environment.
gc() #Frees up the memory
setwd(file.path('kaggle','Titanic-Prediction-Practice'))

dir.create('output', showWarnings = FALSE)
dir.create('rscript', showWarnings = FALSE)
dir.create('tmp', showWarnings = FALSE)

##Libraries
library(randomForest)
library(dplyr)

##
train <- read.csv("./input/train.csv",na.strings=c('NA',''))
test  <- read.csv("./input/test.csv",na.strings=c('NA',''))

train$memo <- 'train'
test$memo <- 'test'
test$Survived <- NA

total <- rbind(train,test)

total$FamilySize <- (total$Parch + total$SibSp + 1)
total$Title <- sapply(total$Name,function(x) substr(x,gregexpr(', ',x)[[1]][1]+2,gregexpr('. ',x,fixed=T)[[1]][1]-1))
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
FamilySizeD <- NA
total$FamilySizeD[total$FamilySize == 1] <- 'singleton'
total$FamilySizeD[total$FamilySize <= 4 & total$FamilySize > 1] <- 'small'
total$FamilySizeD[total$FamilySize > 4] <- 'large'
total$FamilySizeD <- factor(total$FamilySizeD)

variable.names <- c('Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','Mother','Child','FamilySizeD') 
##Find out which col has NA values.
tmp <- apply(total,2,function(x) sum(is.na(x),na.rm=T)) %>% data.frame
tmp <- rownames(tmp)[which(tmp[,1]>0)]
##col "Age" has NA values.

#Remove rows with NA value.
#train[complete.cases(train),] %>% nrow
#train <- train[complete.cases(train),]

##Predict age to replace NA values.
##Maybe I can use for loop and eval to replace NA values automatically
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

train <- total[total$memo=="train",]
test <- total[total$memo=="test",]

prediction.tv <-"Survived"
rf.call <- paste0("randomForest(as.factor(",prediction.tv,")~",variable.names %>% paste0(collapse="+"),", data=train, importance=TRUE, proximity=TRUE, ntree=2000)")
titanic.rf <- eval(parse(text=rf.call))
#titanic.rf <- randomForest(as.factor(Survived)~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked, data=train, importance=TRUE, proximity=TRUE, ntree=2000)
#varImpPlot(titanic.rf)
test$Survived <- predict(titanic.rf, test)

#library(party)
#titanic.cf <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilySizeD + Mother + Child, data = train, controls=cforest_unbiased(ntree=2000, mtry=3))

#test$Survived <- predict(titanic.cf, test, OOB=TRUE, type = "response")
prediction.output <- test[,c("PassengerId","Survived")]
write.csv(prediction.output,'output\\test_prediction.csv',row.names=F)


####
importance    <- importance(titanic.rf)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
library(ggplot2)
library(ggthemes)
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()
