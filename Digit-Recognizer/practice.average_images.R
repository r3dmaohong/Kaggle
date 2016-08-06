##https://www.kaggle.com/c/digit-recognizer/prospector#70

##rm(list = ls()) #Remove all objects in this environment.
gc() #Frees up the memory

##Libraries
library(png)
library(data.table)

setwd(file.path("Kaggle","Digit-Recognizer"))

dir.create('output', showWarnings = FALSE)
dir.create('rscript', showWarnings = FALSE)
dir.create('tmp', showWarnings = FALSE)

##averages the images, for uses as prospect
train <- fread("input/train.csv")
setDF(train)

labels <- as.numeric(train[,1])
train <- train[,-1]

bigify <- function(im, r){
    result <- matrix(nrow=nrow(im)*r, ncol=ncol(im)*r)
    for(i in 1:nrow(im)){
        for(j in 1:ncol(im)){
            for(k in 1:r){
                for(l in 1:r){
                    result[(i - 1) * r + (k - 1) + 1, (j - 1) * r + (l - 1) + 1] <- im[i, j]
                }
            }
        }
    }
    return(result)
}

images <- list()
for(x in 0:9){
    average <- colMeans(train[labels == x,])
    im <- t(matrix(1 - average/256, nrow = 28))
    im <- bigify(im, 8)
    images[[as.character(x)]] <- im
}

row1 <- matrix(nrow=28*8)
for(i in 0:4){
    row1 <- cbind(row1, images[[as.character(i)]])
}
row1 <- row1[,-1] # kills the NA row

row2 <- matrix(nrow=28*8)
for(i in 5:9){
    row2 <- cbind(row2, images[[as.character(i)]])
}
row2 <- row2[,-1] # kills the NA row

writePNG(rbind(row1, row2), "output\\averages.png")
