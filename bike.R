rm(list = ls())
#load data
bike_data= read.csv("/home/akshay/Downloads/day.csv")
dim(bike_data)
str(bike_data)
summary(bike_data)
#check missing values in data
sum(is.na(bike_data))
#no missing values
# dropping instant as it is just a index, dropping dteday as month,year,day info is givem 
# is already available, dropping casual and registered as cnt is sum of both casual & registered
bike_data = subset(bike_data,select = -c(instant,dteday,casual,registered))
#categorical variables
categoricalvar = c("season","yr","mnth","holiday","weekday","workingday","weathersit")
#numerical variables
numericalvar = c("temp","atemp","hum","windspeed")
#visualisation
for(i in categoricalvar){
  dev.new()
  barplot(table(bike_data[,i]),col = "red",main = i,xlab = i,ylab = "values", cex.lab = 1,cex.axis = 1)
}
for(i in numericalvar){
  dev.new()
  hist(bike_data[,i],col = "violet",main = i,xlab = i,ylab="values",breaks = 30,cex.lab = 1,cex.axis = 1)
}
#bike count a/c to season
for(i in numericalvar){
  dev.new()
  plot(bike_data[,i],bike_data$cnt,col = "red",xlab = i,ylab = "bike count")
}
#outlier analysis
library(ggplot2)
cvariables= colnames(bike_data[,c("temp","atemp","windspeed","hum")])
for (i in 1:length(cvariables))
{
  assign(paste0("gn",i), ggplot(aes_string(y = cvariables[i]), data = bike_data)+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="blue", fill = "orange" ,outlier.shape=15,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cvariables[i])+
           ggtitle(paste("Box plot for",cvariables[i])))
}
gridExtra::grid.arrange(gn1,gn3,gn2,gn4,ncol=2)

cor(bike_data$windspeed,bike_data$cnt)
cor(bike_data$hum,bike_data$cnt)
#outliers found in windspeed and hum
#Remove outliers in Windspeed
rem = bike_data$windspeed[bike_data$windspeed %in% boxplot.stats(bike_data$windspeed)$out]
bike_data = bike_data[which(!bike_data$windspeed %in% rem),]

#Remove outliers in hum
rem = bike_data$hum[bike_data$hum %in% boxplot.stats(bike_data$hum)$out]
bike_data = bike_data[which(!bike_data$hum %in% rem),]

#correlation analysis
cor(bike_data)
bike_data

correlation_matrix = cor(bike_data[,numericalvar])
corrplot(correlation_matrix,method = "number",type = 'lower')
#temp and atemp are highly correlated

#Anova
for(i in categoricalvar){
  print(i)
  aov_summary = summary(aov(bike_data$cnt~bike_data[,i],data = bike_data))
  print(aov_summary)
  
}
#dropping atemp,weekday,workingday,holiday 
bike_data  = subset(bike_data,select = -c(temp,weekday,workingday,holiday))
bike_data
#linear regression
require(caTools)

set.seed(166)

sample = sample.split(bike_data$cnt, SplitRatio = .80)
train = subset(bike_data, sample == TRUE)
test  = subset(bike_data, sample == FALSE)
dim(train)
dim(test)
linearmodel = lm(cnt~.,train)
summary(linearmodel)
predictmodel = predict(linearmodel,test[,1:8])
# Calculating Mean Absolute Percent Error(MAPE)
install.packages("Metrics")
library(Metrics)
mape(test[,9],predictmodel) 
# Calculating Mean Absolute Error(MAE)
mae(test[,9],predictmodel) 
rmse(test$cnt,predictmodel) 

#decision tree
decisiontree = rpart(cnt ~ .,train)
summary(decisiontree)
preddecision = predict(decisiontree, test[,1:8])
mape(test[,9],preddecision ) 
mae(test[,9],preddecision )
rmse(test$cnt,preddecision )

#randomforest
library("dplyr")
library("rpart")
library("randomForest")
randommodel = randomForest(cnt ~., train)
summary(randommodel)
randompredict = predict(rf_model, test_data[,1:8])
mape(test[,9],randompredict ) 
mae(test[,9],randompredict )
rmse(test$cnt,randompredict )


