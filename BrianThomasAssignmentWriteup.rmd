---
title: "Machine Learning Assignment"
author: "Brian Thomas"
date: "February 12, 2016"
output:
  pdf_document:
    fig_caption: yes
    keep_tex: yes
    latex_engine: xelatex
    toc: yes
    toc_depth: 2
  html_document:
    toc: yes
    toc_depth: 2
---

###Thanks!
####First, let us thank the contributors of this dataset for their conscientious work in compiling the data, and their generosity in making it available to the public:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Now, let's get this analysis going...

###Libraries
Load some useful libraries

```{r, message=FALSE, comment="> ", warning=FALSE, cache=TRUE}
library(caret)
library(dplyr)
library(data.table)
library(ggplot2)
library(rpart)
library(doParallel)
library(knitr)
```

###Get the Data
Open a connection to the data, download the data, and close the connection.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}

urltr<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
contr<-url(urltr, open = "rb" )
trdat<-fread(urltr, na.strings="NA")
trdat<-tbl_df(trdat)
close(contr)
set.seed(1991)


urlte<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
conte<-url(urlte, open="rb")
tedat<-fread(urlte, na.strings="NA")
tedat<-tbl_df(tedat)
close(conte)
rm(contr,conte, urlte, urltr)

```

Create a data partition to separate training and cross validation data sets.  Once these are created, we'll get a feel for the dimensions.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
inTrain<-createDataPartition(y=trdat$classe, p=.70, list=FALSE)
trdat1<-trdat[inTrain,]
cvdat<-trdat[-inTrain,]
rm(inTrain)
dim(trdat1)
dim(cvdat)
dim(tedat)
```

###Clean the data
Let's look at the class of each variable
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
sapply(trdat1, class) 
```

We find numerous "Character" classes that should be numeric for our analysis.
Let's change these
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
trdat1<-cbind(trdat1[,1:6], sapply(trdat1[,7:159], as.numeric), trdat1[,160])
```

There are NA values dispersed throughout the dataset.  Let's change these to zero
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
trdat1[is.na(trdat1)]<-0
```

There are also #DIV/0! values.  Let's try to change these to zero as well.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
trdat1[which(trdat1=="#DIV/0!")]<-0
```


See how many zeroes exist in each variable after transformation.  If the number is a strong majority, the variable may not be very useful.  We will return a vector of column indices.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
zero_df<-data.frame()
for(i in 1:160){
  zero_df[i,1]<-i
  zero_df[i,2]<-sum(trdat1[,i]==0)
}
zero_df
```

There are a lot of variables containing very little data.  
Let's isolate those that are mostly zero and remove them from each dataset
Also, it's time to remove column "V1" since it is going to introduce problems later.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
zero_cols<-zero_df[which(zero_df[,2]>5000),1]; zero_cols;
trdat2<-trdat1[,-c(1,zero_cols)]; dim(trdat2)
cvdat2<-cvdat[,-c(1,zero_cols)]; dim(cvdat2)
tedat2<-tedat[,-c(1,zero_cols)]; dim(tedat2)
```

###Choose a Model
My initial plan is to run an "rf" model.  I have set up the following trainControl parameters.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
tc<-trainControl(method = "oob",
                 number = 10,
                 repeats = 10,
                 p = 0.75,
                 search = "grid",
                 initialWindow = NULL,
                 horizon = 1,
                 fixedWindow = TRUE,
                 verboseIter = FALSE,
                 returnData = TRUE,
                 returnResamp = "final",
                 savePredictions = FALSE,
                 classProbs = TRUE,
                 summaryFunction = defaultSummary,
                 selectionFunction = "best",
                 preProcOptions = list(thresh = 0.95, ICAcomp = 3, k = 5),
                 sampling = NULL,
                 index = NULL,
                 indexOut = NULL,
                 timingSamps = 0,
                 predictionBounds = rep(FALSE, 2),
                 #seeds = seeds,
                 adaptive = list(min = 5, alpha = 0.05,
                                 method = "gls", complete = TRUE),
                 trim = FALSE,
                 allowParallel = TRUE)
```

After numerous failed attempts at training models due to memory limits,
I decided to randomly split my data into several models.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
dat_size<-dim(trdat2)[1]/3
dat_size2<-dat_size+1
dat_size3<-dat_size*2
dat_size4<-dat_size3+1
dat_size5<-dat_size*3
samp_vect<-sample(dim(trdat2)[1],dim(trdat2)[1],replace = FALSE)
trdat2_1<-trdat2[samp_vect[1:dat_size],]; dim(trdat2_1)
trdat2_2<-trdat2[samp_vect[(dat_size2):(dat_size3)],]; dim(trdat2_2)
trdat2_3<-trdat2[samp_vect[(dat_size4):(dat_size5)],]; dim(trdat2_3)
```

```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
tgrid<- expand.grid(mtry = 50)
memory.limit(size=10000000000000)

registerDoParallel(4)
memory.limit(size=10000000000000)
t1<-Sys.time()
modFit_1<-train(classe~., 
              trdat2_1, 
              method = "rf",  
              metric = "Accuracy",   
              maximize = TRUE,
              trControl = tc, 
              tuneGrid = tgrid,
              tuneLength=50,
              prox=TRUE
)
t2<-Sys.time()
Mod1_time<-t2-t1

registerDoParallel(4)
memory.limit(size=10000000000000)
t1<-Sys.time()
modFit_2<-train(classe~., 
              trdat2_2, 
              method = "rf",  
              metric = "Accuracy",   
              maximize = TRUE,
              trControl = tc, 
              tuneGrid = tgrid,
              tuneLength=50,
              prox=TRUE
)
t2<-Sys.time()
Mod2_time<-t2-t1

registerDoParallel(4)
memory.limit(size=10000000000000)
t1<-Sys.time()
modFit_3<-train(classe~., 
              trdat2_3, 
              method = "rf",  
              metric = "Accuracy",   
              maximize = TRUE,
              trControl = tc, 
              tuneGrid = tgrid,
              tuneLength=50,
              prox=TRUE
)
t2<-Sys.time()
Mod3_time<-t2-t1
```

I decided to capture the time it took to run each model for general interest.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
Mod1_time
Mod2_time
Mod3_time
```

###Results
Here are the model results for the training and cross-validation sets.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
modFit_1
modFit_2
modFit_3

acc_1<-confusionMatrix(predict(modFit_1, cvdat2), cvdat2$classe);acc_1
acc_2<-confusionMatrix(predict(modFit_2,cvdat2), cvdat2$classe); acc_2
acc_3<-confusionMatrix(predict(modFit_3,cvdat2), cvdat2$classe);acc_3
```

All that is left is to predict against the test set.
```{r, message=FALSE, echo=TRUE, comment="> ", warning=FALSE, cache=TRUE, results='asis', include=TRUE}
predict(modFit_1, tedat2)
predict(modFit_2, tedat2)
predict(modFit_3, tedat2)
```

###Final Thoughts
Each model performs admirably well against the test data.

There are a number of additional steps and considerations that should be made along the way.  I was not fond of removing the columns of data to simplify the analysis; however, in this case, I believe the results warranted the action.  While I performed plenty of exploratory analysis, it is not included here.  I encountered numerous outliers, but testing the algorithms showed that even with outliers, the models predicted at an accurate level.