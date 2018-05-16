library(caret)
library(dplyr)
library(mlbench)

# Load data
setwd(getwd())
states.data <- readRDS("states.rds")

# Look at the data
head(states.data)

# Predictions Goal: csat score
# Remove vsat and msat since the sum of both gives the prediction we want.
states.data$vsat <- NULL
states.data$msat <- NULL

# Prepare training scheme
control <- trainControl(method="cv", number=5)

# Omit state variable
states.data.filter <- states.data %>% select(-state)
# Remove rows with NAs
states.data.filter <- states.data.filter[complete.cases(states.data.filter),]
summary(states.data.filter)

# train the LM model
set.seed(27)
model_Lm <- train(csat~.,data=states.data.filter,method="lm",trControl=control)
summary(model_Lm)

# train the GLMNET model
set.seed(27)
model_Glmnet <- train(csat~.,data=states.data.filter,method="glmnet",tuneGrid = expand.grid(alpha=0:1,lambda = seq(0.0001,1,length=20)),trControl=control)
plot(model_Glmnet)

# train the SVM model
set.seed(27)
model_Svm <- train(csat~., data=states.data.filter, method="svmRadial", trControl=control)
summary(model_Svm)

# collect resamples
results <- resamples(list(LM=model_Lm,SVM=model_Svm,GLMNET = model_Glmnet))
# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)
# dot plots of results
dotplot(results)





