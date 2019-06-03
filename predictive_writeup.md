---
title: "Prediction Assignment Writeup"
author: "Saifeel Momin"
date: "5/29/2019"
output: 
  html_document:
    keep_md: true 
        
---

# 1. Introduction 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The training and testing data sets can be found here: 

**Training:** https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

**Testing:** https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# 2. Data Analysis and Exploratory Analysis 

We begin by loading in the data sets, packages, and setting the seed to ensure reproducibility 

```r
library(caret)
library(ggplot2)
library(data.table)
library(parallel)
library(doParallel)
library(rattle)
library(rpart.plot)
set.seed(41444)

main <- read.csv("pml-training.csv")
TEST <- read.csv("pml-testing.csv")

dim(main)
```

```
## [1] 19622   160
```

```r
dim(TEST)
```

```
## [1]  20 160
```

```r
head(colnames(main))
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"
```

```r
tail(colnames(main))
```

```
## [1] "accel_forearm_y"  "accel_forearm_z"  "magnet_forearm_x"
## [4] "magnet_forearm_y" "magnet_forearm_z" "classe"
```
Now using the caret package we will parition the training data set into a training and testing. We'll use a ratio of .65(training) to .35(testing). The goal of our predictive model is to be able to predict the 'classe' variable as accuaretly based on the provided data. The classe variable determines the correctness of a specific kind of excercise performed. Class A is assigned for exercises performed correctly and the rest of the classes B - E are assigned when the excercise is not performed correctly. 


```r
##creating test and train sets 
inTrain <- createDataPartition(y = main$X, p = .65, list = FALSE)
training <- main[inTrain, ]
test <- main[-inTrain,]
dim(training)
```

```
## [1] 12756   160
```

```r
dim(test)
```

```
## [1] 6866  160
```
The training and testing sets have 160 variables that we can use in our models. However, we will test the variables and will select only those which provide us with some predicitvie strength for our models. We'll begin by testing for near zero variance. 


```r
n <- nearZeroVar(training)
training2 <- training[,-n]
test2 <- test[,-n]
n
```

```
##  [1]   6  12  13  14  15  16  17  20  23  26  51  52  53  54  55  56  57
## [18]  58  59  69  70  71  72  73  74  75  81  82  87  88  89  90  91  92
## [35]  95  98 101 125 126 127 128 129 130 133 136 139 143 144 145 146 147
## [52] 148 149 150
```
The output from calling nearZeroVar is saved into 'n' and the variables with near zero variance are removed from the train and test sets as they provide no benefits for our models. Additionally, reducing the overall number of variables will enable our train() call to run more efficiently and accuaretly. 

Next, we will remove all columns with a high proportion of NA values in addition to removing classifier variables. The first 5 columns are all classifier variables and could potentially mislead our models. (1 - X, 2 - user_name, 3- raw_timestamp_part_1, 4- raw_timestamp_part2, 5 - cvtd_timestamp)


```r
##removing vars with high NA value percentage and classifiers vars
NAs <- sapply(training2, function(x) mean(is.na(x))) > 0.95
training3 <- training2[, NAs==FALSE]
test3 <- test2[, NAs==FALSE]

##removing classifier vars 
train0 <- training3[,-(1:5)]
test0 <- test3[,-(1:5)]
x <- train0[,-54]
y <- train0[,54]
dim(train0)
```

```
## [1] 12756    54
```

```r
dim(test0)
```

```
## [1] 6866   54
```
The train and test sets now have a significantly reduced number of variables and areready to be utilized in the building of our predictive model. The overall goal of our model is to predict the 'classe' variable which correlates to the correctness of exercise. 

# 3. Prediction Model Building

Before we start building our model we will intitialize clusters and parallel processing to speed up the processing of our train() calls. Following this,  we'll set the trainControl parameters and create our model fit. We will use a random forest model and a generalized boosted model. Once both models are created we will determine which has the greatest accuracy and lowest out-of-bag (OOB) error rate. The best model will then be applied to the TEST data set. 

### a. Random forests model 

```r
##clusters for faster processing 
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)
set.seed(41444)
# rf model fit 
fitControl <- trainControl(method = "cv", number = 4, verboseIter = FALSE, allowParallel = TRUE)
modFit <- train(x,y, data = train0, method = "rf", trControl = fitControl)
stopCluster(cluster)
registerDoSEQ()
#prediction on test dataset 
modPredict <- predict(modFit, newdata= test0)
predictMat <- confusionMatrix(modPredict, test0$classe)
predictMat
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1940    3    0    0    0
##          B    0 1349    3    0    2
##          C    0    1 1185    4    0
##          D    0    1    0 1088   11
##          E    0    0    0    0 1279
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9964          
##                  95% CI : (0.9946, 0.9976)
##     No Information Rate : 0.2826          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9954          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9963   0.9975   0.9963   0.9899
## Specificity            0.9994   0.9991   0.9991   0.9979   1.0000
## Pos Pred Value         0.9985   0.9963   0.9958   0.9891   1.0000
## Neg Pred Value         1.0000   0.9991   0.9995   0.9993   0.9977
## Prevalence             0.2826   0.1972   0.1730   0.1590   0.1882
## Detection Rate         0.2826   0.1965   0.1726   0.1585   0.1863
## Detection Prevalence   0.2830   0.1972   0.1733   0.1602   0.1863
## Balanced Accuracy      0.9997   0.9977   0.9983   0.9971   0.9950
```
The Random forests model seems to perform well as the accuracy of predictions is 99.62%.


### b. Generalized boosted model 


```r
#gbm model fit
fitControl1 <- trainControl(method = "repeatedcv", number = 5, repeats=1, verboseIter = FALSE)
modFit1 <- train(classe~., data = train0, method = "gbm", trControl = fitControl1, verbose = FALSE )
#prediction on test dataset
modPredcitGBM <- predict(modFit1, newdata = test0)
predictMatGBM <- confusionMatrix(modPredcitGBM, test0$classe)
predictMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1930   13    0    0    0
##          B   10 1328    8    8    1
##          C    0   11 1173   17    4
##          D    0    2    5 1065   19
##          E    0    0    2    2 1268
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9851         
##                  95% CI : (0.982, 0.9879)
##     No Information Rate : 0.2826         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9812         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9948   0.9808   0.9874   0.9753   0.9814
## Specificity            0.9974   0.9951   0.9944   0.9955   0.9993
## Pos Pred Value         0.9933   0.9801   0.9734   0.9762   0.9969
## Neg Pred Value         0.9980   0.9953   0.9974   0.9953   0.9957
## Prevalence             0.2826   0.1972   0.1730   0.1590   0.1882
## Detection Rate         0.2811   0.1934   0.1708   0.1551   0.1847
## Detection Prevalence   0.2830   0.1973   0.1755   0.1589   0.1853
## Balanced Accuracy      0.9961   0.9879   0.9909   0.9854   0.9904
```

The Generalized boosted model performs well with an accuracy of 98.78% for its predictions. We'll compare the accuracy of both prediction models and determine which one should be applied to the main test dataset. 


```r
rfplot <- plot(predictMat$table, col = predictMat$byClass, main = paste("Random Forest - Accuracy =", round(predictMat$overall['Accuracy'], 4)))
```

![](predictive_writeup_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
gmbplot <- plot(predictMatGBM$table, col = predictMatGBM$byClass, main = paste("Generalized Boosted Model - Accuracy =", round(predictMatGBM$overall['Accuracy'], 4)))
```

![](predictive_writeup_files/figure-html/unnamed-chunk-7-2.png)<!-- -->


The random forest models performs slightly better than the generalized boosted model and so we will the rf model for our main test dataset. 

# 4. Applying model to Test Data 

```r
testPredict <- predict(modFit, newdata = TEST)
testPredict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
