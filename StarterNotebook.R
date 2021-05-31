# =====================================================================================================================
#
# Yassir ETA prediction - UmojaHack SA
#
# Megan Beckett                                
# https://www.linkedin.com/in/meganbeckett/     
# https://twitter.com/mbeckett_za               
#
# =====================================================================================================================

# This is based on the data from https://zindi.africa/hackathons/umojahack-south-africa-yassir-eta-prediction-challenge/data.

# STAGES --------------------------------------------------------------------------------------------------------------

# 1. Acquire data.
# 2. Load data.
# 3. Wrangle data.
# 4. Exploratory analysis.
# 5. Build models.


# ACQUIRE DATA --------------------------------------------------------------------------------------------------------

# 1. Create an account on Zindi.
# 2. Login to Zindi and join the hackathon.
# 3. Navigate to the URL above.
# 4. Download the data.
# 5. Unpack the ZIP archive into a data/ folder.


# LIBRARIES -----------------------------------------------------------------------------------------------------------
library(caret)               # Swiss Army Knife for ML

library(readr)               # Reading CSV files
library(dplyr)               # General data wrangling
library(janitor)             # Cleaning column names
library(ggplot2)             # Wicked plots
library(lubridate)           # Handling date/time data


# LOAD DATA -----------------------------------------------------------------------------------------------------------
PATH_TRAIN <- file.path("data", "Train.csv")
PATH_TEST <- file.path("data", "Test.csv")
PATH_WEATHER <-file.path("data", "Weather.csv")

# Read in the data.
trips <- read_csv(PATH_TRAIN)
weather <- read_csv(PATH_WEATHER)

# Take a look at the column names.
names(trips)
names(weather)

# Improve column names (using snake case).
trips <- trips %>% clean_names()
names(trips)

# Look at structure of data.
str(trips)

# Look at a "spreadsheet" view of data.
View(trips)

# We can immediately drop the ID column since this cannot have any predictive value.
trips <- trips %>% select(-id)


# WRANGLE -------------------------------------------------------------------------------------------------------------
# Create a date from the timestamp for trips to be able to join the weather data by date
trips <- trips %>%
  mutate(date = as.Date(timestamp))


# FEATURE ENGINEERING -------------------------------------------------------------------------------------------------
# Convert the trip start numeric time of day (perhaps time of day plays a part, for example during peak traffic hours)
trips <- trips %>% 
  mutate(time_of_day = lubridate::hour(timestamp)
         )


# WEATHER DATA --------------------------------------------------------------------------------------------------------
# Add in the weather data for each trip according to its date of departure
trips <- trips %>% 
 left_join(weather, by = "date")


# EDA: PLOTS ----------------------------------------------------------------------------------------------------------
# ETA compared to trip distance - there could be a linear relationship
ggplot(trips, aes(x = trip_distance, y = eta)) +
  geom_point(alpha = 0.1) +
  labs(x = "Trip distance (m)", y = "ETA (seconds)")

# Distribution of eta - is it normally distributed?
ggplot(trips, aes(x = eta)) +
  geom_histogram()

# Distribution of trip distance
ggplot(trips, aes(x = trip_distance)) +
  geom_histogram()

# Distribution of rain fall
ggplot(trips, aes(x = total_precipitation)) +
  geom_histogram()

# For interest, let's visualise where these trips are mostly departing from.
# Perhaps there is a city/urban/rural factor to take into account later
# We will plot this with the leaflet library
library(leaflet)

leaflet(trips) %>% 
  addTiles() %>% 
  addMarkers(
    lng = ~origin_lon,
    lat = ~origin_lat,
    clusterOptions = markerClusterOptions()
)


# TRAIN/TEST SPLIT ----------------------------------------------------------------------------------------------------

# In order to assess how well our model performs we need to split it into two components:
#
# - training and
# - testing.
#
# There are a number of ways to do this split.

# Set the RNG seed so that everybody gets the same train/test split.
#
set.seed(13)

# Generally you want to have around 80:20 split.
#
index <- sample(c(TRUE, FALSE), nrow(trips), replace = TRUE, prob = c(0.8, 0.2))

train <- trips[index,]
test <- trips[!index,]

# Check that the proportions are correct (should be roughly 4x as many records in training data).
#
nrow(train)
nrow(test)


# MODEL: LM ------------------------------------------------------------------------------------------------------------
# We’ll kick off by looking at a simple linear regression model. And just throw (almost) everything in - the "kitchen sink" approach.
trips_lm <- lm(eta ~ 
               origin_lat + origin_lon +
               destination_lat + destination_lon +
               trip_distance + 
               time_of_day +
               # Weather data 
               dewpoint_2m_temperature + maximum_2m_air_temperature + mean_2m_air_temperature +
               mean_sea_level_pressure + minimum_2m_air_temperature + surface_pressure +
               total_precipitation + u_component_of_wind_10m + v_component_of_wind_10m,
               data = train)

summary(trips_lm)

# What about some interactions between different weather recordings for the day, for example rain and wind?
trips_lm <- update(trips_lm, . ~ . + total_precipitation:u_component_of_wind_10m)
#
summary(trips_lm)

# Make predictions on the testing data.
#
test_predictions <- predict(trips_lm, test)
head(test_predictions)
#
# So how well does the model perform? Need to compare the known to the predicted classes.
#
head(test$eta)
#
# The standard approach to evaluating a linear regression model is to calculate the RMSE. 
# Calculate the RMSE.
rmse(test$eta, test_predictions)

# This is a rather large RMSE. 


# MODEL: KNN -----------------------------------------------------------------------------------------------------------
# Next, we look at another technique which is not a linear model but simple and powerful for regression: 
# k-Nearest Neighbours (kNN). 
# The principle is simple: assign a value derived from a collection of “nearby” observations.
# This technique is very flexible (it works well for both classification and regression problems).
library(kknn)

trips_knn <- kknn(eta ~
                    origin_lat + origin_lon +
                    destination_lat + destination_lon +
                    trip_distance + 
                    time_of_day +
                    # Weather data 
                    dewpoint_2m_temperature + maximum_2m_air_temperature + mean_2m_air_temperature +
                    mean_sea_level_pressure + minimum_2m_air_temperature + surface_pressure +
                    total_precipitation + u_component_of_wind_10m + v_component_of_wind_10m,
                  train, test, k = 7, kernel = "optimal") 

# Calculate the RMSE.
rmse(test$eta, predict(trips_knn))

# This is an improvement on the lm model.

# MODEL: CARET / DECISION TREE ----------------------------------------------------------------------------------------

# Next, let's try a decision tree using caret.
trips_rpart <-  train(eta ~
                        origin_lat + origin_lon +
                        destination_lat + destination_lon +
                        trip_distance + 
                        time_of_day +
                        # Weather data 
                        dewpoint_2m_temperature + maximum_2m_air_temperature + mean_2m_air_temperature +
                        mean_sea_level_pressure + minimum_2m_air_temperature + surface_pressure +
                        total_precipitation + u_component_of_wind_10m + v_component_of_wind_10m,
                    data = train, method = "rpart",
                    trControl = trainControl("cv", number = 10),
                    tuneLength = 10
)

trips_rpart

test_predictions <- predict(trips_rpart, test)

rmse(test$eta, test_predictions)

# The RMSE is still large. A potential problem is over fitting.


# MODEL: RANDOM FORESTS --------------------------------------------------------------------------------------------
# Random forests improve predictive accuracy by generating a large number of bootstrapped trees 
# (based on random samples of variables), classifying a case using each tree in this new "forest",
# and deciding a final predicted outcome by combining the results across all of the trees (an average in 
# regression, as we are doing here).
library(randomForest)

# This will take a little longer to run!
trips_rf <- randomForest(eta ~
                           origin_lat + origin_lon +
                           destination_lat + destination_lon +
                           trip_distance + 
                           time_of_day +
                           # Weather data 
                           dewpoint_2m_temperature + maximum_2m_air_temperature + mean_2m_air_temperature +
                           mean_sea_level_pressure + minimum_2m_air_temperature + surface_pressure +
                           total_precipitation + u_component_of_wind_10m + v_component_of_wind_10m,
                         data = train)

test_predictions <- predict(trips_rf, test)

rmse(test$eta, test_predictions)


# MODEL: BOOSTING ------------------------------------------------------------------------------------------------
# Boosting is another approach to improve the predictions resulting from a decision tree.
# In the Random Forests bagging approach, each tree is built upon a bootstrapped data set, independent of other
# trees, and then combined to build a single model. In boosting, each tree is built sequentially so that 
# each one is grown using information from the previous. 
#
# Let's try it out with our data!
library(gbm)

trips_boost = gbm(eta ~
                    origin_lat + origin_lon +
                    destination_lat + destination_lon +
                    trip_distance + 
                    time_of_day +
                    # Weather data 
                    dewpoint_2m_temperature + maximum_2m_air_temperature + mean_2m_air_temperature +
                    mean_sea_level_pressure + minimum_2m_air_temperature + surface_pressure +
                    total_precipitation + u_component_of_wind_10m + v_component_of_wind_10m, 
                  data = train, distribution = "gaussian", 
                  n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)

test_predictions <- predict(trips_boost, test, n.trees = 10000)

rmse(test$eta, test_predictions)


# NEXT STEPS ----------------------------------------------------------------------------------

# Each of the above models can be improved. These provide a starting point. 
#
# For example, they can be improved through:
#   - Feature selection (are all variables needed or necessary?)
#   - Feature engineering (can you create new variables that have strong predictive power? Perhaps
#     interactions between different weather metrics, ie. maybe if there is strong wind and precipitation
#     on a day. How can ou create a variable to combine this and test this hypothesis)
#   - Parameter tuning (this is particularly relevant for the last models which have many parameters 
#     which can be further tuned)


# SUBMISSION ----------------------------------------------------------------------------------

# Once you have your final model you are happy with, the nest step is to make predictions on the unseen
# testing set.

# We will call this unseen data test_final to distinguish from the testing data used during model building and training.

# Read in the final test data
test_final <- read_csv(PATH_TEST) %>% clean_names()

# Perform the same data preparation steps as was done during training
test_final <- test_final %>%
  mutate(date = as.Date(timestamp)) %>% 
  mutate(time_of_day = lubridate::hour(timestamp)) %>%
  left_join(weather, by = "date")

# Make predictions using your final model. We will use the Random Forest model for demonstration purposes here.
final_predictions <- predict(trips_rf, test_final)

final_predictions <- as.data.frame(final_predictions)

# Create the submission file
submission <- test_final %>%
  bind_cols(final_predictions) %>%
  select(ID = id, ETA = final_predictions)

# Write as csv
write_csv(submission, "data/submission_yassir.csv")


