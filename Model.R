### Generating the Data

## Loading required Libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

## Downloading and Wrangling the Raw Data
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

## Creating the Validation set using 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#Subset the edx set into training and test sets
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

#Remove excess objects and files
rm(dl, ratings, movies, test_index, temp, movielens, removed)
unlink(file.path(getwd(),"ml-10M100K"), recursive = TRUE)

### Building the Model

## Writing the function to calculate the RMSE
RMSE <- function(x,y){
  sqrt(mean((x-y)^2))
}

##Initialising Data Frame for the Model Results

#Predicting using only the mean
mu <- mean(train$rating)
naive_rmse <- RMSE(train$rating, mu)

#Results data frame
rmse_results <- data_frame(Method = "Just the average", RMSE = naive_rmse, Lamda = NA)

## Modelling the Movie Effect
l <- 5

b_i <- train %>% group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+l), n = n())

b_i %>% ggplot(aes(b_i)) + geom_histogram(bins = 15, color = "black")

b_i %>% sample_n(30) %>%
  ggplot(aes(as.character(movieId), b_i, size=n)) + 
  geom_point() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + xlab("movieId")

#Finding optimal Lamda Parameter
l <- seq(0,20,0.25)
RMSE_lamda <- map_df(l,function(l){

  b_i <- train %>% group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  predicted_ratings <-  test %>% left_join(b_i, by = "movieId") %>%
    mutate(pred = mu + b_i) %>% pull(pred)
  
  data.frame(l = l, RMSE = RMSE(predicted_ratings,test$rating))
})

#Visualizing the effect of Lamda
RMSE_lamda %>% ggplot(aes(l,RMSE)) + geom_point()

#Inputing results
result <- min(RMSE_lamda$RMSE)
l <- RMSE_lamda$l[which.min(RMSE_lamda$RMSE)]
rmse_results <- bind_rows(rmse_results, data_frame(Method = "Mean + b_i", RMSE = result, Lamda = l))

#Saving the optimal Movie Effect
b_i <- train %>% group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + l))

## Modelling the User Effect

# Analysing the user Effect with an arbitrary lambda
l<- 5

b_u <- train %>% left_join(b_i, by = "movieId") %>% group_by(userId) %>% 
  summarize(b_u = sum(rating - mu - b_i)/(n()+l), n = n())

b_u %>% sample_n(30) %>% 
  ggplot(aes(as.character(userId), b_u, size = n, alpha = 0.3)) + 
  geom_point() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust =1)) + 
  xlab("userId")

b_u %>% ggplot(aes(b_u)) + geom_histogram(bins = 15, color = "black")


#Finding optimal Lamda Parameter
l <- seq(0,20,0.25)
RMSE_lamda <- map_df(l,function(l){

  b_u <- train %>% left_join(b_i, by = "movieId") %>% group_by(userId) %>% 
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
  predicted_ratings <-  test %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>% pull(pred)
  
  data.frame(l = l, RMSE = RMSE(predicted_ratings,test$rating))
})

#Visualizing the effect of Lamda
RMSE_lamda %>% ggplot(aes(l,RMSE)) + geom_point()

#Inputing reults
result <- min(RMSE_lamda$RMSE)
l <- RMSE_lamda$l[which.min(RMSE_lamda$RMSE)]
rmse_results <- bind_rows(rmse_results, data_frame(Method = "Mean + b_i + b_u", RMSE = result, Lamda = l))

#Saving Optimal User effect
b_u <- train %>% left_join(b_i, by = "movieId") %>% group_by(userId) %>% 
  summarize(b_u = sum(rating - mu - b_i)/(n() + l))

## Modelling the Genre Effect

#Exploration of the genre effect with an arbitrary Lamda
l <- 5

b_g <- train %>% left_join(b_i, by= "movieId") %>% left_join(b_u, by = "userId") %>% 
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l), n = n()) 

#Visualising the Genre effect

b_g %>% sample_n(20) %>% ggplot(aes(genres, b_g, size = n, alpha = 0.3)) + 
  geom_point() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

b_g %>% ggplot(aes(b_g)) + geom_histogram(bins = 15, color = "black")

#Finding optimal Lamda Parameter
l <- seq(0,10,0.25)
RMSE_lamda <- map_df(l,function(l){
  
  b_g <- train %>% left_join(b_i, by= "movieId") %>% left_join(b_u, by = "userId") %>% group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))
  
  predicted_ratings <-  test %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>% pull(pred)
  
  data.frame(l = l, RMSE = RMSE(predicted_ratings,test$rating))
})

#Visualizing the effect of Lamda
RMSE_lamda %>% ggplot(aes(l,RMSE)) + geom_point()


#Inputing reults
result <- min(RMSE_lamda$RMSE)
l <- RMSE_lamda$l[which.min(RMSE_lamda$RMSE)]
rmse_results <- bind_rows(rmse_results, data_frame(Method = "Mean + b_i + b_u + b_g", RMSE = result, Lamda = l))

#Saving Optimal Genre effect
b_g <- train %>% left_join(b_i, by= "movieId") %>% left_join(b_u, by = "userId") %>% group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))

##Modeling a more Detailed Genre-effect
#Seperating the genres variable into its components in both sets
train <- train %>% separate_rows(genres, sep = "\\|") 
test <- test %>% separate_rows(genres, sep = "\\|")

#Exploration of the genre effect with an arbitrary Lamda
l <- 5

b_g <- train %>% left_join(b_i, by= "movieId") %>% left_join(b_u, by = "userId") %>% 
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l), n = n())

#Visualising the new Genre effect
b_g %>% ggplot(aes(genres, b_g, size = n)) + geom_point() + theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Visualising the distribution
b_g %>% ggplot(aes(b_g)) + geom_histogram(color = "black")

#Model
l <- seq(0,20,0.25)
RMSE_lamda <- map_df(l,function(l){

  b_g <- train %>% left_join(b_i, by= "movieId") %>% left_join(b_u, by = "userId") %>% group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))
  
  predicted_ratings <-  test %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>% pull(pred)
  
  data.frame(l = l, RMSE = RMSE(predicted_ratings,test$rating))
})

#Visualizing the effect of Lamda
RMSE_lamda %>% ggplot(aes(l,RMSE)) + geom_point()

#Inputing reults
result <- min(RMSE_lamda$RMSE)
l <- RMSE_lamda$l[which.min(RMSE_lamda$RMSE)]
rmse_results <- bind_rows(rmse_results, data_frame(Method = "mu + b_i + b_u + detailed_b_g", RMSE = result, Lamda = l))

#Saving Optimal Genre effect
b_g <- train %>% left_join(b_i, by= "movieId") %>% left_join(b_u, by = "userId") %>% group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))

## Modeling the effect of the release year

#Extracting the release of the Movie from the title
train <- train %>% 
  mutate(year = str_extract(title,"\\(\\d{4}\\)")) %>% 
  mutate(year = str_replace_all(year, "[//(//)]", ""))

test <- test %>%
  mutate(year = str_extract(title,"\\(\\d{4}\\)")) %>% 
  mutate(year = str_replace_all(year, "[//(//)]", ""))

#Exploring the Release year effect with an arbitrary Lamda
l <- 5

b_y <- train %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by = "genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u - b_g)/(n()+l), n = n())

b_y %>% ggplot(aes(as.numeric(year), b_y, size = n, alpha = 0.3)) + 
  geom_point() + xlab("year")

b_y %>% ggplot(aes(as.numeric(year),y = b_y)) + 
  geom_smooth() + geom_point() + xlab("year")

b_y %>% ggplot(aes(b_y)) + geom_histogram(color = "black")

#Model
l <- seq(0,20,0.25)
RMSE_lamda <- map_df(l,function(l){
  
  b_y <- train %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by = "genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u - b_g)/(n()+l))
  
  predicted_ratings <-  test %>% 
    left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by = "genres") %>% left_join(b_y, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_y) %>% pull(pred)
  
  data.frame(l = l, RMSE = RMSE(predicted_ratings,test$rating))
})

#Visualizing the effect of Lamda
RMSE_lamda %>% ggplot(aes(l,RMSE)) + geom_point()

#Inputing reults
result <- min(RMSE_lamda$RMSE)
l <- RMSE_lamda$l[which.min(RMSE_lamda$RMSE)]
rmse_results <- bind_rows(rmse_results, data_frame(Method = "mu + b_i + b_u + detailed_b_g + b_y", RMSE = result, Lamda = l))

#Saving Optimal release year effect
b_y <- train %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by = "genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u - b_g)/(n()+l))

#Final Model

Model <- function(validation){
  validation <- validation %>% separate_rows(genres, sep = "\\|")
  validation <- validation %>% 
    mutate(year = str_extract(title,"\\(\\d{4}\\)")) %>% 
    mutate(year = str_replace_all(year, "[//(//)]", ""))
  
  validation %>% left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    left_join(b_y, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_y) %>% pull(pred)
}  


RMSE_final <- function(validation,pred){
  validation <- validation %>% separate_rows(genres, sep = "\\|")
  validation <- validation %>% 
    mutate(year = str_extract(title,"\\(\\d{4}\\)")) %>% 
    mutate(year = str_replace_all(year, "[//(//)]", ""))
  
  RMSE(validation$rating,pred)
  
}

pred <- Model(validation)

RMSE_final(validation,pred)
