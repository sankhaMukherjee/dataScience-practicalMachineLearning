# Predicting training efficiency

> **Synopsis**: Data collected from devices such as the *Jawbone Up*, *Nike FuelBand* etc., is used to determine if particular exercises are done properly or not. There is separate data available for the training of various machine learning algorithms, and separate ones for testing the efficiency of the different training methodologies used. In this study, we shall see whether it might be possible to determine, given some sets of data, whether a set of exercises were performed correctly. We just need to determine if barbell lifts were performed correctly. 

## 1. Data Exploration

The data is obtained from the website on [Human Activity Recognition][1]. A couple of datasets have been selected from this website, and provided in the problem statement. These are the [pml-training.csv][2] and [pml-testing.csv][3] sets respectively. These have been individually downloaded and put into the current working folder. 

### 1.1. Loading and cleaning the data

Since there is a lot of data, we want to be training our datasets solely on the data which is available, it is more prodent to get rid of data which has only NA's. Unfortunately, in the data, some missing values are defined as `""`, some as `"#DIV/0!"` and some are defined as `NA`. Hence, we need o properly ascertain the `NA` values. We simply assign all of these as `NA` as we load the dataset. Next, any column that has any `NA` is simply dropped, since, we dont know how a particular learning algorithm will deal with missing data. Also the first 7 columns (like time, and row numbers) are very possibly not very good predictors. So, it might just be better to get rid of them. 


```r
training <- read.csv('pml-training.csv', na.strings=c("NA","","#DIV/0!"))
dropNa   <- complete.cases(t(training))
training <- training[, dropNa]
training <- training[,8:length(colnames(training))]
names(training)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

Now we have a set of 51 predictors, and the `"classe"` variable which we wish to predict. *Note that we shall later be using crossvalidation*. So at this point, there is not real need for creating a separate dataset for training and a separate one for validation.

### 1.3. Feature Definition

Now that we have a *complete* dataset, we want see of some of the columns are covariates. Note that we dont include the variable we want to predict here ...


```r
library(ggplot2)
library(RColorBrewer)
myPalette   <- colorRampPalette(brewer.pal(8,"Spectral"))(100)
corrMatrix  <- cor(as.matrix( training[  1:(length(colnames(training)) -1 )  ] ))
heatmap(corrMatrix, col=myPalette)
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2-1.png) 

There does appear to be some form of correlation. Hence, we shall use some preconditioning using PCA. Also, we shall be crossvalidating our entire dataset. This will allow us to not have to deal with such things as creating extra training and validatin sets. Since we have so much data, we will simply do a 10 fold cross validation. 


```r
tc <- trainControl(method = "cv", number = 5, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)
```

## 2. Training

For training, we shall use a number of models. These are:

| model          |  model type                                               |
|---------------:|:----------------------------------------------------------|
| "rf"           |  random forests                                           |
| "svmLinear"    |  linear support vector machines                           |
| "svmRadial"    |  radial support vector machines                           |
| "nnet"         |  neural networks                                          |
| "bayesglm"     |  bayesian generalized linear models                       |
| "LogitBoost"   |  a statistical implementation for the Ada boost algorithm.|
 
Note that in this case, we shall make absolutely no attempt at optimizing and fine tuning the performance of any of the models. We shall use whatever comes out-of-the-box, so that we can finish this assignment in a reasonable amount of time. Furthermore, all the models use the same training conditions of cross validation and feature selection. This will put some algorithms at a disadvantage. However, we shall not dwell on such theoretical possibilities too much, especially since we have very little time to complete this assignment. 

 

```r
library(kernlab)
library(arm)
## This is going to take for ever. Its much better to do it over night. Time to sleep.
models <- c("rf", "svmRadial", "nnet", "svmLinear", "bayesglm","LogitBoost")
output <- lapply( models , function(m) {train(classe ~ ., data = training, method = m, trControl= tc)} )
```



```r
#output[[1]]$results$Kappa
maxVals   <- sapply( output, function(m){ max(m$results$Accuracy) } )
kappaVals <- sapply( output, function(m){ max(m$results$Kappa) } )

as.data.frame(cbind(models, maxVals, kappaVals))
```

```
##       models           maxVals         kappaVals
## 1         rf 0.994088237012797 0.992521767809257
## 2  svmRadial 0.934003814378902 0.916358226017652
## 3       nnet  0.43904680524463 0.292402594071221
## 4  svmLinear 0.785292799977285 0.726997605855878
## 5   bayesglm 0.402812684588007 0.236225152292079
## 6 LogitBoost   0.8996493501289 0.872218581641724
```

As is evident, some models are good, while others are not so good. 

## 3. Prediction

Before prediction, we have to make sure that we have the same type of dataframe for the testing set. Loading the testing dataset. We will only keep the columns which are present in the training dataset. The rest of the columns are uselesss. 


```r
testing <- read.csv('pml-testing.csv', na.strings=c("NA","","#DIV/0!"))
testing1 <- testing[,names(training)[1:52]]
```

Now that we have the entire set of data, we can examine what each of our models predict for the testing dataset sets. 


```r
prediction <- sapply( output, function(m) { predict(m, newdata = testing1) } )
prediction <- as.data.frame(prediction)
names(prediction) <- models
prediction
```

```
##    rf svmRadial nnet svmLinear bayesglm LogitBoost
## 1   B         B    C         C        B          B
## 2   A         A    B         A        A          A
## 3   B         B    B         B        B          A
## 4   A         A    A         C        B          C
## 5   A         A    D         A        B          A
## 6   E         E    D         E        B          E
## 7   D         D    D         D        B          D
## 8   B         B    B         B        B          D
## 9   A         A    A         A        A          A
## 10  A         A    A         A        A          A
## 11  B         B    C         C        B       <NA>
## 12  C         C    B         A        A       <NA>
## 13  B         B    D         B        B          B
## 14  A         A    A         A        A          A
## 15  E         E    D         E        B       <NA>
## 16  E         E    D         E        B          E
## 17  A         A    A         A        A          A
## 18  B         B    B         B        B          B
## 19  B         B    B         B        B       <NA>
## 20  B         B    D         B        B          B
```

As can be see, every model makes a different prediction. Which one is correct? There is no right answer here. Although it is possible to come up with different stakhing possibilities, over here, we simply find the values which are predicted by most of the learning algorithms. That is, a simple majority of the different algorithms is what we are going to [choose][4].


```r
possibilities = c('A', 'B', 'C', 'D', 'E')
counter        <- sapply(possibilities, function(m){rowSums(prediction == m, na.rm = T)})
finalSelection <- colnames(counter)[apply(counter,1,which.max)] 
counter <- as.data.frame(counter)
counter['finalSelection'] = finalSelection
counter['rf'] <- prediction['rf']
counter
```

```
##    A B C D E finalSelection rf
## 1  0 4 2 0 0              B  B
## 2  5 1 0 0 0              A  A
## 3  1 5 0 0 0              B  B
## 4  3 1 2 0 0              A  A
## 5  4 1 0 1 0              A  A
## 6  0 1 0 1 4              E  E
## 7  0 1 0 5 0              D  D
## 8  0 5 0 1 0              B  B
## 9  6 0 0 0 0              A  A
## 10 6 0 0 0 0              A  A
## 11 0 3 2 0 0              B  B
## 12 2 1 2 0 0              A  C
## 13 0 5 0 1 0              B  B
## 14 6 0 0 0 0              A  A
## 15 0 1 0 1 3              E  E
## 16 0 1 0 1 4              E  E
## 17 6 0 0 0 0              A  A
## 18 0 6 0 0 0              B  B
## 19 0 5 0 0 0              B  B
## 20 0 5 0 1 0              B  B
```

How does it compare with the *best* model (i.e. the random fores)? It turns out that the `finalSelection` criterion is *identical* to the return value of random forest except for prediction for observation number 12. Let us examine it a little more closely. For prediction number 12, both random forest and svm radial determine that the answer should be `C`, while, the bayesian glm and logistic boosting methods both choose `A`. Note that the predictibility of both logistic boost (0.8722)  and bayesian glm (0.24) is much worse than the random forest (0.9925) and radial svm (0.916). So my guess is that, even though the `finalSelection` selects `A`, the answer should really be `C`. For all other selections, the result should be relatively safe. 

## 4. Writing output files

The files are now written with the different values in them for submission. 


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(finalSelection)
```


## 5. Results and Discussion

After computing the final selection, and submitting the results, it appears that all the results are correct, except number 12. This was already something which we thought may be the case. In fact, it turns out that this is indeed the case. In all probability, the result for question 12 should have been `C` and not `A`.

[1]:http://groupware.les.inf.puc-rio.br/har
[2]:https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
[3]:https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
[4]:http://stackoverflow.com/questions/17735859/for-each-row-return-the-column-name-of-the-largest-value"
