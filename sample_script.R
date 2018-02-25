# Load libraries

library(data.table)
library(randomForest)
library(magrittr)
library(dplyr)
library(caret)
library(xgboost)
library(revgeo)
library(nnet)

# Load data
path = ""
train <- fread(paste0(path,"train.csv"))
train <- data.frame(train)
test <- fread(paste0(path,"test.csv"))
test <- data.frame(test)
ids = test$id
test$log_price <- NA

all_data <- data.frame(rbind(train,test))
train_set = 1:nrow(train)
test_set <- (nrow(train)+1):(nrow(train) + nrow(test))

# Select a subset of the data

keep_cols <- c('bedrooms', 'beds', 'property_type','room_type','bed_type','cancellation_policy','city',
               'accommodates','bathrooms','latitude','longitude',
               'number_of_reviews','review_scores_rating','log_price')

all_data <- all_data[,keep_cols]

# Impute missing values with mean of column

fillna <- function(column) {
  column[is.na(column)] <- mean(column, na.rm = TRUE)
  return(column)
}

zip <- revgeo(longitude = all_data$longitude[1:10], latitude = all_data$latitude[1:10], 
              output = 'hash', item = 'zip');
all_data$zip <- unlist(zip)


col_type <- sapply(all_data,class)
numeric_type <- !(col_type %in% c("character","factor"))
all_data[,numeric_type] <- sapply(all_data[,numeric_type], fillna)
all_data <- all_data %>% mutate_if(is.character,as.factor)


# split train and validate

train <- all_data[train_set,]
test <- all_data[test_set,]

set.seed(2453)
trainIndex <- createDataPartition(train$log_price, p = .75, 
                                  list = FALSE, 
                                  times = 1)
train_test <- train[-trainIndex,]
train <- train[trainIndex,]

# rmse function
rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}


# Train a Random Forest model with cross-validation

# cv_folds <- sample(1:3, size = nrow(train), replace = TRUE)
# 
# for(i in 1:3) {
#   # Train the model using the training sets
#   fit <- randomForest(log_price ~ .,
#                       data = train[cv_folds !=i,],
#                       ntree = 10)
#   
#   # Make predictions using the testing set
#   preds <- predict(fit, train[cv_folds == i,])
#   
#   # Calculate RMSE for current cross-validation split
#   print(mean((preds - train[cv_folds == i,'log_price'])^2)^.5)
# }

# knn
train <- all_data[train_set,]
train_test <- train[-trainIndex,]
train <- train[trainIndex,]
train <- train %>% select(bedrooms, beds, accommodates, bathrooms, 
                          number_of_reviews, review_scores_rating, 
                          log_price, latitude, longitude, room_type)


# tc <- trainControl("cv", 10, savePredictions=T)  #"cv" = cross-validation, 10-fold
tunegrid <- expand.grid(.k=c(1:4))
fit <- train(log_price ~ ., preProcess = c('center','scale'), data = train,
             method = "knn", tunegrid = tunegrid)


rmse(train_testY, predict(fit, train_testX))




train <- train %>% select(bedrooms, beds, accommodates, bathrooms, 
                          number_of_reviews, review_scores_rating, log_price)

# glm

tc <- trainControl("cv", 10)  #"cv" = cross-validation, 10-fold
fit <- train(log_price ~ ., preProcess = c('center','scale'), data = train,
             method = "glm", trControl = tc, metric = 'RMSE')

rmse(train_test$log_price, predict(fit, select(train_test, -log_price)))

# random forest

train <- all_data[train_set,]
train <- train %>% select(bedrooms, beds, accommodates, bathrooms, 
                          number_of_reviews, review_scores_rating, 
                          log_price, latitude, longitude, room_type, property_type, 
                          cancellation_policy)
train$room_type <- as.factor(train$room_type)
train$property_type <- as.factor(train$property_type)
train$cancellation_policy <- as.factor(train$cancellation_policy)

train_test <- train[-trainIndex,]
train <- train[trainIndex,]


tc <- trainControl("cv", 2)  #"cv" = cross-validation, 10-fold
tunegrid <- expand.grid(.mtry=c(1:12))
fit2 <- train(log_price ~ ., data = train, method = "rf", 
              metric = 'RMSE', tunegrid = tunegrid, trControl = tc)

rmse(train_test$log_price, predict(fit2, select(train_test, -log_price)))

# xgboost

train <- all_data[train_set,]
train <- train %>% select(bedrooms, beds, accommodates, bathrooms, 
                          number_of_reviews, review_scores_rating, 
                          log_price, latitude, longitude, room_type, property_type, 
                          cancellation_policy)
train$room_type <- as.factor(train$room_type)
train$property_type <- as.factor(train$property_type)
train$cancellation_policy <- as.factor(train$cancellation_policy)

train_test <- train[-trainIndex,]
train <- train[trainIndex,]


tc <- trainControl("cv", 10)  #"cv" = cross-validation, 10-fold
tuneGridXGB <- expand.grid(
  nrounds=c(350),
  lambda = c(.01, .1),
  alpha = c(.01, .1),
  eta = c(0.05, 0.1))


fit3 <- train(log_price ~ ., data = train, method = "xgbLinear",
              metric = 'RMSE', tunegrid = tuneGridXGB, trControl = tc)


rmse(train_test$log_price, predict(fit3, select(train_test, -log_price)))


# neural net - 1-layer

tc <- trainControl("cv", 10)  #"cv" = cross-validation, 10-fold
tuneGridNN <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))

fit4 <- train(log_price ~ ., data = train, method = "nnet", maxit = 1000, 
              tuneGrid = tuneGridNN, trControl = tc, trace = F, linout = TRUE) 

rmse(train_test$log_price, predict(fit4, select(train_test, -log_price)))


# neural net - multi-layer

# tc <- trainControl("cv", 10)  #"cv" = cross-validation, 10-fold
# tuneGridMNN <- expand.grid(.layer1 = c(5, 6, 7), .layer2 = c(5, 6, 7), .layer3 = c(5, 6, 7),
#                            learning.rate = c(.05, .1), dropout = c(0.2,0.6), activation = 'relu')
tc <- trainControl("cv", 1)  #"cv" = cross-validation, 10-fold
tuneGridMNN <- expand.grid(.layer1 = c(5), .layer2 = c(1), .layer3 = c(0),
                           learning.rate = c(.1), dropout = c(0.2), activation = 'relu')

fit5 <- train(log_price ~ ., data = train, method = "mxnet", 
              tuneGrid = tuneGridMNN, trControl = tc) 

rmse(train_test$log_price, predict(fit5, select(train_test, -log_price)))

# Create submission file

prediction <- predict(fit3, test)

sample_submission <- data.frame(id = ids, log_price = prediction)
write.csv(sample_submission, "submission3.csv", row.names = FALSE)
