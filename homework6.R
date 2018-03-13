# read data into memory
training_digits <- read.csv("hw06_mnist_training_digits.csv", header = FALSE)
training_labels <- read.csv("hw06_mnist_training_labels.csv", header = FALSE)
test_digits <- read.csv("hw06_mnist_test_digits.csv", header = FALSE)
test_labels <- read.csv("hw06_mnist_test_labels.csv", header = FALSE)

# get X and y values
Xtrain <- as.matrix(training_digits) / 255
ytrain <- training_labels[,1]
Xtest <- as.matrix(test_digits) / 255
ytest <- test_labels[,1]

# get number of samples and number of features
N <- length(ytrain)
D <- ncol(Xtrain)

# function to get the most occuring


# calculate class means 
classmeans <- NULL
for (i in 1:10){
  csum <-0
  for (xi in 1:length(training_labels[,])){
    if(training_labels[xi,]==i){
      csum<- csum + Xtrain[xi,]
    }
  }
  classmeans<- rbind(classmeans,csum)
}
classmeans <- classmeans/50

# calculate within class scatter matrix
S_W<-0
for (r in 1:10){
  S_I<-0
  for (i in 1:length(training_labels[,])){
    if(training_labels[i,]==r){
      S_I <-S_I +   as.matrix((Xtrain[i,]-classmeans[r,])) %*% t(as.matrix((Xtrain[i,]-classmeans[r,])))
      
    }
  }
  S_W <- S_W + S_I
}

# calculate between class scatter matrix
S_B <- 0
totalmean <- colSums(Xtrain[,])/500
for (i in 1:10){
  temp <-    50 *as.matrix(classmeans[i,]-totalmean) %*% t(as.matrix(classmeans[i,]-totalmean)) 
  S_B <- S_B +temp
}

# handling non-invertable matrix formation
diagonal <-diag(784)*0.0000000001
S_W <- S_W+diagonal

# calculate eigenvalues and eigenvectors
decomposition <- eigen(solve(S_W)%*% S_B, symmetric = TRUE)

# calculate two-dimensional projections
Z <- (Xtrain - matrix(colMeans(Xtrain), N, D, byrow = TRUE)) %*% decomposition$vectors[,1:2]
ZT <- (Xtest - matrix(colMeans(Xtest), N, D, byrow = TRUE)) %*% decomposition$vectors[,1:2]

# plot two-dimensional projections training and test data
point_colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6")
plot(Z[,1], Z[,2], type = "p", pch = 19, col = point_colors[ytrain], cex = 0,
     xlab = "Dimension 1", ylab = "Dimension 2", las = 1,main="Training points")
text(Z[,1], Z[,2], labels = ytrain %% 10, col = point_colors[ytrain])

plot(ZT[,1], ZT[,2], type = "p", pch = 19, col = point_colors[ytest], cex = 0,
     xlab = "Dimension 1", ylab = "Dimension 2", las = 1,main="Test points")
text(ZT[,1], ZT[,2], labels = ytest %% 10, col = point_colors[ytest])

# calculate accuracy for test
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
accuracy <- NULL
for (S in 1:9){
  Ztrain <- (Xtrain - matrix(colMeans(Xtrain), N, D, byrow = TRUE)) %*% decomposition$vectors[,1:S]
  Ztest <- (Xtest - matrix(colMeans(Xtrain), N, D, byrow = TRUE)) %*% decomposition$vectors[,1:S]
  predictions <- NULL
  for(i in 1:500){
    dist<- NULL
    for(j in 1:500){
      dist <-  cbind(dist, (abs(Ztest[i,]-Ztrain[j,])))
    }
    predictions <- c(predictions,getmode(ytest[order((colSums(dist^2))^1/2, decreasing = FALSE)[1:5]]))
  }
  
  counter<-0
  for(a in 1:500){
    if(predictions[a]==ytest[a]){
      counter <- counter +1
    }
  }
  accuracy <- c(accuracy,counter/500)
}

plot(accuracy*100,xlab = "R", ylab = "Classification accuracy(%)",type = "o")