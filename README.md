# kaggle_project2019
kaggle_project_airbnb_price_prediction


library(xgboost) # for xgboost
library(tidyverse) # general utility functions
library(plyr)

setwd('C:/Class/Semester -  1/FRM1/Kaggle')

analysis <- read.csv('AnalysisData.csv')
scoring <- read.csv('ScoringData2.csv')


data <- rbind.fill(analysis, scoring)

str(data)

data <- data %>% select(-c(zipcode, id, name, summary, space, description, neighborhood_overview, notes, transit, access, interaction, house_rules,
                          host_name, host_location, host_about, host_neighbourhood, host_verifications, street, neighbourhood, neighbourhood_group_cleansed, 
                          city, state, market, smart_location, country_code, country, amenities, calendar_updated, license, jurisdiction_names
                          ))


data[data == "N/A"] <- NA
sapply(data, function(x) sum(is.na(x)))

data <- data %>% select(-c(weekly_price, square_feet, monthly_price, host_acceptance_rate, host_response_time, host_response_rate))

str(data)

data$host_since <- as.numeric(as.POSIXct(data$host_since, format="%Y-%m-%d"))
data$first_review <- as.numeric(as.POSIXct(data$first_review, format="%Y-%m-%d"))
data$last_review <- as.numeric(as.POSIXct(data$last_review, format="%Y-%m-%d"))

str(data)

library(data.table)
data <- data %>% select(-c(has_availability, requires_license, is_business_travel_ready))

tf.list <- c('host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'is_location_exact', 'instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification')


data <- data.table(data) %>%
  .[host_is_superhost == "t", host_is_superhost := "1"] %>%
  .[host_is_superhost == "f", host_is_superhost := "0"]

data <- data.table(data) %>%
  .[host_has_profile_pic == "t", host_has_profile_pic := "1"] %>%
  .[host_has_profile_pic == "f", host_has_profile_pic := "0"]

data <- data.table(data) %>%
  .[host_identity_verified == "t", host_identity_verified := "1"] %>%
  .[host_identity_verified == "f", host_identity_verified := "0"]

data <- data.table(data) %>%
  .[is_location_exact == "t", is_location_exact := "1"] %>%
  .[is_location_exact == "f", is_location_exact := "0"]

data <- data.table(data) %>%
  .[instant_bookable == "t", instant_bookable := "1"] %>%
  .[instant_bookable == "f", instant_bookable := "0"]

data <- data.table(data) %>%
  .[require_guest_profile_picture == "t", require_guest_profile_picture := "1"] %>%
  .[require_guest_profile_picture == "f", require_guest_profile_picture := "0"]

data <- data.table(data) %>%
  .[require_guest_phone_verification == "t", require_guest_phone_verification := "1"] %>%
  .[require_guest_phone_verification == "f", require_guest_phone_verification := "0"]


str(data)

neighbourhood_cleansed <- model.matrix(~neighbourhood_cleansed-1, data)
#zipcode <- model.matrix(~zipcode-1, data)
property_type <- model.matrix(~property_type-1, data)
room_type <- model.matrix(~room_type-1, data)
bed_type <- model.matrix(~bed_type-1, data)
cancellation_policy <- model.matrix(~cancellation_policy-1, data)

data <- data %>% select(-c(neighbourhood_cleansed, property_type, room_type, bed_type, cancellation_policy))

data <- cbind(data, neighbourhood_cleansed, property_type, room_type, bed_type, cancellation_policy) 

for (i in tf.list) {
data[[i]] <- as.character(data[[i]])
data[[i]] <- as.integer(data[[i]])
}

str(data)

data <- data %>% select(-i)

dt <- data[complete.cases(data$price), ]
sb <- data[!complete.cases(data$price), ]


# Define Target
x <- dt %>% select(-price)
y <- dt%>%
  select(price)
sb <- sb %>% select(-price)

# convert into matrix
x<- as.matrix(x)
y<- as.matrix(y)
sb<- as.matrix(sb)

#Split dataset into testing and training subsets
#get the numb 70/30 training test split
numberOfTrainingSamples <- round(length(y) * .7)

# training data
train_data <- x[1:numberOfTrainingSamples,]
train_labels <- y[1:numberOfTrainingSamples]

# testing data
test_data <- x[-(1:numberOfTrainingSamples),]
test_labels <- y[-(1:numberOfTrainingSamples)]

str(dt)
#Convert the cleaned dataframe to a dmatrix
# put our testing & training data into two seperates Dmatrixs objects

dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)


#Training our model
model <- xgboost(data = dtrain, # the data 
                 max.depth = 5, # the maximum depth of each decision tree
                 #nfold = 5,
                 nround = 300, # max number of boosting iterations
                 #verbose = FALSE,
                 eval_metric = "rmse",
                 early_stopping_rounds = 30,
                 lambda = 5,
                 #alpha = 10,
                 min_child_weight = 6,
                 #eta = 0.1,
                 gamma = 0.1,
                 subsample=0.9,
                 objective = "reg:gamma")  # the objective function

pred = predict(model,newdata=dtest)

# model assesment
library(Metrics)
rmse(test_labels, pred)




ttrain <- xgb.DMatrix(data = x, label= y)


total <- scoring$id
for (i in (1:10)){
model <- xgboost(data = ttrain, # the data 
                 max.depth = 5, # the maximum depth of each decision tree
                 #nfold = 5,
                 nround = 300, # max number of boosting iterations
                 #verbose = FALSE,
                 eval_metric = "rmse",
                 early_stopping_rounds = 30,
                 lambda = 5,
                 #alpha = 10,
                 min_child_weight = 6,
                 #eta = 0.1,
                 gamma = 0.1,
                 seed = 1,
                 subsample=0.9,
                 objective = "reg:gamma")  # the objective function

pred = predict(model,newdata=sb)
total <- cbind(total, pred)
}

write.csv(total, 'total',row.names = F)
