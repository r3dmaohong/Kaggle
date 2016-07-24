library(reshape2)
library(foreach) 
library(doSNOW)
library(data.table)
##Reference
##https://gist.github.com/jpetterson/776aba5d190d0fe84fdd
##https://www.kaggle.com/c/facial-keypoints-detection/details/getting-started-with-r

##Go parallel
cl<-makeCluster(4) ##4 Cores
registerDoSNOW(cl)

setwd(file.path("Kaggle","Facial-Keypoints-Detection"))

# read data and convert image strings to arrays
d.train <- fread("input\\training.csv", stringsAsFactors=F)
#d.test <- fread(,  stringsAsFactors=F)

set.seed(0)
idxs <- sample(nrow(d.train), nrow(d.train)*0.8)
d.test  <- d.train[-idxs, ]
d.train   <- d.train[idxs, ]


##Split d.train$Image into data frame by whitespace.
##Check: substr(d.train$Image[1],1,7); im.train[1,1:2];
im.train <- foreach(im = d.train$Image, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}
im.test <- foreach(im = d.test$Image, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}
d.train$Image <- NULL
d.test$Image <- NULL
#save.image("dp.RData")

##List the coordinates we have to predict
##Get names with "_x", and remove the word "_x". (Both x and y will have the same names)
coordinate.names <- gsub("_x", "", names(d.train)[grep("_x", names(d.train))])


##For each one, compute the average patch.
##In foreach, if you use function in libraries, you should add it in to the statement. (Ex. ".packages="foreach")
patch_size  <- 10 ##patch_size is the number of pixels which are going to extract in each direction around the center of the keypoint.

mean.patches <- foreach(coord = coordinate.names, .packages="foreach") %dopar% {
  cat(sprintf("computing mean patch for %s\n", coord))
  coord_x <- paste0(coord, "_x")
  coord_y <- paste0(coord, "_y")
  
  ##Compute average patch.
  patches <- foreach (i = 1:nrow(d.train), .combine=rbind) %dopar% {
    im  <- matrix(data = im.train[i,], nrow=96, ncol=96) ##96*96 = 9216 (im.train[i,]'s length)
    x   <- d.train[i, coord_x]
    y   <- d.train[i, coord_y]
    #A square of 21x21 pixels (10+1+10).
    x1  <- (x-patch_size)
    x2  <- (x+patch_size)
    y1  <- (y-patch_size)
    y2  <- (y+patch_size)

    if((!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96)){
      as.vector(im[x1:x2, y1:y2])
    }else{
      NULL
    }
  }
  matrix(data = colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)
}
##If using 1:21, images will be reversed.
#image(1:21, 1:21, mean.patches[[1]][21:1,21:1], col=gray((0:255)/255))
#image(1:21, 1:21, mean.patches[[3]][21:1,21:1], col=gray((0:255)/255))
#save.image("aft-mean.patches.RData")

##For each coordinate and for each test image, find the position that best correlates with the average patch.

#######
search_size <- 2 ##Indicates how many pixels we are going to move in each direction when searching for the keypoint. We will center the search on the average keypoint location, and go search_size pixels in each direction.
p <- foreach(coord_i = 1:length(coordinate.names), .combine=cbind, .packages="foreach") %dopar% {
  ##The coordinates want to predict
  coord <- coordinate.names[coord_i]
  coord_x <- paste(coord, "x", sep="_")
  coord_y <- paste(coord, "y", sep="_")
  
  ##The average of the col in the training set.
  mean_x  <- mean(d.train[, coord_x], na.rm=T)
  mean_y  <- mean(d.train[, coord_y], na.rm=T)
  
  ##Search space: 'search_size' pixels centered on the average coordinates 
  x1 <- as.integer(mean_x)-search_size
  x2 <- as.integer(mean_x)+search_size
  y1 <- as.integer(mean_y)-search_size
  y2 <- as.integer(mean_y)+search_size
  
  ##Ensure only consider patches completely inside the image.
  x1 <- ifelse(x1-patch_size<1, patch_size+1, x1)
  y1 <- ifelse(y1-patch_size<1, patch_size+1, y1)
  x2 <- ifelse(x2+patch_size>96, 96-patch_size, x2)
  y2 <- ifelse(y2+patch_size>96, 96-patch_size, y2)
  
  ##Build a list of all positions to be tested.
  params <- expand.grid(x = x1:x2, y = y1:y2)
  
  ##For each image...
  r <- foreach(i = 1:nrow(d.test), .combine=rbind) %dopar% {
    if((coord_i==1)&&((i %% 100)==0)){ 
      cat(sprintf("%d/%d\n", i, nrow(d.test))) 
    }
    im <- matrix(data = im.test[i,], nrow=96, ncol=96)
    
    ##Compute a score for each position ...
    r  <- foreach(j = 1:nrow(params), .combine=rbind) %dopar% {
      x <- params$x[j]
      y <- params$y[j]
      p <- im[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size)]
      score <- cor(as.vector(p), as.vector(mean.patches[[coord_i]]),method="spearman")
      score <- ifelse(is.na(score), 0, score)
      data.frame(x, y, score)
    }
    
    # ... and return the best
    best <- r[which.max(r$score), c("x", "y")]
  }
  names(r) <- c(coord_x, coord_y)
  r
}
#######

##Prepare file for submission
predictions <- data.frame(ImageId = 1:nrow(d.test), p)
##Transfet table format.
submission <- melt(predictions, id.vars="ImageId", variable.name="FeatureName", value.name="Location")
##Submissin format
example.submission <- read.csv('input\\IdLookupTable.csv'))
sub.col.names      <- c("RowId","Location")
example.submission$Location <- NULL

str(submission)
str(example.submission)
submission <- merge(example.submission, submission, all.x=T, sort=F)
submission <- submission[, sub.col.names]

stopCluster(cl)

write.csv(submission, "output\\submission_search.csv", quote=F, row.names=F)
