#.79909 Top 20% Kaggle

#Titanic
library(ggplot2); library(tidyverse);library(caret);library(randomForest); library(caTools)

setwd("G:/Personal/Kaggle/Titanic")

############################################################################################ Data Processing
#Import tidydata
rawdata <- read.csv2("train.csv", header = TRUE, sep = ',', stringsAsFactors = FALSE)

tidydata <- rawdata
#tidydata <- tidydata[,c(1:3,5:8,10)]
tidydata$Age <- as.numeric(tidydata$Age)
tidydata$Survived <- as.factor(tidydata$Survived)
tidydata$Fare <- as.numeric(tidydata$Fare)

############################################################################################ Exploratory Analysis
with(tidydata,table(Sex, Survived))

ageSurvival <- tidydata %>%
  select(Age,Survived,Sex) %>%
  mutate(
    ageGroup =ifelse(Age <=12,'Child',ifelse(Age > 18,'Adult',"Teenager")))
with(ageSurvival,table(ageGroup,Survived))

ggplot(ageSurvival, aes(x=Age, y=Survived, color = Sex))+
  geom_point(shape = 1)+
  ggtitle('Survival By Age, Gender')+
  xlab('Age') + 
  ylab('Survived')+
  theme_classic()

tidydata$Pclass <- as.factor(tidydata$Pclass)
ggplot(tidydata,aes(x = Age, y=Survived, color = Pclass))+
  geom_point()+
  ggtitle('Survival by Age,Class')+
  xlab('Age')+
  ylab('Survived')+
  theme_classic()

tidydata$Pclass <- as.factor(tidydata$Pclass)
ggplot(tidydata,aes(x = Age, y=round(Fare,0), color = Survived))+
  geom_point()+
  ggtitle('Survival by Fare')+
  xlab('Fare')+
  ylab('Survived')+
  theme_classic()



############################################################################################ Feature Engineering
#Sex, turning into numeric so I can model it
modelData <- tidydata
modelData$Female <- ifelse(modelData$Sex == "male",0,1) #Male = 0, female =1
modelData <- modelData[-5] # Drop Sex column since we created a female column

#Embarked column - Turn it into numeric
modelData$Embarked_C <- ifelse(modelData$Embarked == "C",1,0)
modelData$Embarked_S <- ifelse(modelData$Embarked == "S",1,0)
modelData <- modelData[-11]

#Name column - Get title, then turn into numeric column
modelData$Title <- sapply(strsplit(as.character(modelData$Name), " "), `[`, 2)
modelData$Title_mr <- ifelse(modelData$Title == "Mr.",1,0)
modelData$Title_mrs <- ifelse(modelData$Title == "Mrs.",1,0)
modelData$Title_miss <- ifelse(modelData$Title == "Miss.",1,0)
modelData$Title_master <- ifelse(modelData$Title == "Master.",1,0)
modelData$Title_don <- ifelse(modelData$Title == "Don.",1,0)
modelData$Title_rev <- ifelse(modelData$Title == "Rev.",1,0)
modelData$Title_dr <- ifelse(modelData$Title == "Dr.",1,0)
modelData$Title_mme <- ifelse(modelData$Title == "Mme.",1,0)
modelData$Title_ms <- ifelse(modelData$Title == "Ms.",1,0)
modelData$Title_major <- ifelse(modelData$Title == "Major.",1,0)
modelData$Title_mlle <- ifelse(modelData$Title == "Mlle.",1,0)
modelData$Title_col <- ifelse(modelData$Title == "Col.",1,0)
modelData$Title_capt <- ifelse(modelData$Title == "Capt.",1,0)
modelData$Title_jonkheer <- ifelse(modelData$Title == "Jonkheer",1,0)
modelData <- modelData[-4]
modelData <- modelData[-13]

#PClass Variable
modelData$FirstClass <- ifelse(modelData$Pclass == "1",1,0)
modelData$SecondClass <- ifelse(modelData$Pclass == "2",1,0)
modelData <- modelData[-3]

#Cabin Variable
modelData$CabinLevel <- substr(modelData$Cabin, 0, 1)
modelData$ClassA <- ifelse(modelData$CabinLevel=="A",1,0)
modelData$ClassB <- ifelse(modelData$CabinLevel=="B",1,0)
modelData$ClassC <- ifelse(modelData$CabinLevel=="C",1,0)
modelData$ClassD <- ifelse(modelData$CabinLevel=="D",1,0)
modelData$ClassE <- ifelse(modelData$CabinLevel=="E",1,0)
modelData$ClassF <- ifelse(modelData$CabinLevel=="F",1,0)
modelData$ClassG <- ifelse(modelData$CabinLevel=="G",1,0)
modelData <- modelData[-8]
modelData <- modelData[-27]
modelData <- modelData[-33]


#Impute Age variable before creating any features from it
library(mice)
set.seed(123456)
imputed_Data <- mice(modelData, m=1, maxit = 50, method = 'pmm', seed = 500)
modelData <- complete(imputed_Data,1)

#Age Variable #Left out teenager
modelData$Child <- ifelse(modelData$Age <= 13, 1,0)
modelData$Adult <- ifelse(modelData$Age >18,1,0)
modelData <- modelData[-3]

#Create family size variable
modelData$FamilySize <- modelData$SibSp + modelData$Parch

#Drop the Ticket variable
modelData <- modelData[-5]
modelData <- modelData[-3]
modelData <- modelData[-3]

# #Look at fare and class
# fareClass <- modelData %>%
#   select(Fare, FirstClass, SecondClass)


#Check correlations, and remove highly correlated
cor(modelData[-2]) #nothing highly correlated

#Check for near-zero variances, and remove zero variances and near zero variances
# nsv <- nearZeroVar(modelData[,-2], saveMetrics = TRUE)
# nsv <- nearZeroVar(modelData)
# modelData <- modelData[,-nsv]

############################################################################################ Create Training & Test Set
set.seed(123456)
# modelData$Survived <- as.numeric(modelData$Survived)
split <- sample.split(modelData$Survived,SplitRatio = 0.75)
training_set <- subset(modelData,split == TRUE)
test_set <- subset(modelData,split == FALSE)
# training_set$Survived <- as.factor(training_set$Survived)
# test_set$Survived <- as.factor(test_set$Survived)
############################################################################################ Modeling

#See if we can pear down model and still keep accuracy
fit.forest <- randomForest(Survived ~., data = training_set, importance = TRUE, ntree = 250)
varImpPlot(fit.forest, type = 1)
varImp <- importance(fit.forest)
#Remove less important variables, according to VarImp
selVars <- names(sort(varImp[,1], decreasing = TRUE))[1:20]

modRF <- randomForest(x = training_set[,selVars], y = training_set$Survived,
                      ntree = 100,
                      nodesize = 7,
                      importance = TRUE)

# Predicting the Test set results
y_pred = predict(modRF, newdata = as.matrix(test_set[selVars]))

cm = table(test_set[, 2], y_pred)
#84.3

############################################################################################ Kaggle Dataset
#Import the data
rawtestdata <- read.csv2("test.csv", header = TRUE, sep = ',', stringsAsFactors = FALSE)

kaggledata <- rawtestdata
kaggledata$Age <- as.numeric(kaggledata$Age)
# kaggledata$Survived <- as.factor(kaggledata$Survived)
kaggledata$Fare <- as.numeric(kaggledata$Fare)

#Sex, turning into numeric so I can model it
kaggledata$Female <- ifelse(kaggledata$Sex == "male",0,1) #Male = 0, female =1
# kaggledata <- kaggledata[-5] # Drop Sex column since we created a female column

#Embarked column - Turn it into numeric
kaggledata$Embarked_C <- ifelse(kaggledata$Embarked == "C",1,0)
kaggledata$Embarked_S <- ifelse(kaggledata$Embarked == "S",1,0)
# kaggledata <- kaggledata[-11]

#Name column - Get title, then turn into numeric column
kaggledata$Title <- sapply(strsplit(as.character(kaggledata$Name), " "), `[`, 2)
kaggledata$Title_mr <- ifelse(kaggledata$Title == "Mr.",1,0)
kaggledata$Title_mrs <- ifelse(kaggledata$Title == "Mrs.",1,0)
kaggledata$Title_miss <- ifelse(kaggledata$Title == "Miss.",1,0)
kaggledata$Title_master <- ifelse(kaggledata$Title == "Master.",1,0)
kaggledata$Title_don <- ifelse(kaggledata$Title == "Don.",1,0)
kaggledata$Title_rev <- ifelse(kaggledata$Title == "Rev.",1,0)
kaggledata$Title_dr <- ifelse(kaggledata$Title == "Dr.",1,0)
kaggledata$Title_mme <- ifelse(kaggledata$Title == "Mme.",1,0)
kaggledata$Title_ms <- ifelse(kaggledata$Title == "Ms.",1,0)
kaggledata$Title_major <- ifelse(kaggledata$Title == "Major.",1,0)
kaggledata$Title_mlle <- ifelse(kaggledata$Title == "Mlle.",1,0)
kaggledata$Title_col <- ifelse(kaggledata$Title == "Col.",1,0)
kaggledata$Title_capt <- ifelse(kaggledata$Title == "Capt.",1,0)
kaggledata$Title_jonkheer <- ifelse(kaggledata$Title == "Jonkheer",1,0)
# kaggledata <- kaggledata[-4]
# kaggledata <- kaggledata[-13]

#PClass Variable
kaggledata$FirstClass <- ifelse(kaggledata$Pclass == "1",1,0)
kaggledata$SecondClass <- ifelse(kaggledata$Pclass == "2",1,0)
# kaggledata <- kaggledata[-3]

#Cabin Variable
kaggledata$CabinLevel <- substr(kaggledata$Cabin, 0, 1)
kaggledata$ClassA <- ifelse(kaggledata$CabinLevel=="A",1,0)
kaggledata$ClassB <- ifelse(kaggledata$CabinLevel=="B",1,0)
kaggledata$ClassC <- ifelse(kaggledata$CabinLevel=="C",1,0)
kaggledata$ClassD <- ifelse(kaggledata$CabinLevel=="D",1,0)
kaggledata$ClassE <- ifelse(kaggledata$CabinLevel=="E",1,0)
kaggledata$ClassF <- ifelse(kaggledata$CabinLevel=="F",1,0)
kaggledata$ClassG <- ifelse(kaggledata$CabinLevel=="G",1,0)
# kaggledata <- kaggledata[-8]
# kaggledata <- kaggledata[-27]
# kaggledata <- kaggledata[-33]


#Impute Age variable before creating any features from it
library(mice)
set.seed(123456)
imputed_Data_kaggle <- mice(kaggledata, m=1, maxit = 50, method = 'pmm', seed = 500)
kaggledata <- complete(imputed_Data_kaggle,1)

#Age Variable #Left out teenager
kaggledata$Child <- ifelse(kaggledata$Age <= 13, 1,0)
kaggledata$Adult <- ifelse(kaggledata$Age >18,1,0)
# kaggledata <- kaggledata[-3]

#Create family size variable
kaggledata$FamilySize <- kaggledata$SibSp + kaggledata$Parch

# #Drop the Ticket variable
# kaggledata <- kaggledata[-5]
# kaggledata <- kaggledata[-3]
# kaggledata <- kaggledata[-3]

kaggledata <- kaggledata %>%
  select(PassengerId, FamilySize,Female,SecondClass,Fare,Title_mr,Title_miss,Title_mrs,FirstClass,Child,ClassD,
         Title_master,Title_rev,ClassE,ClassC,Embarked_C,Adult,ClassF,Title_dr,Title_don,Title_mme)

y_pred_kaggle = predict(modRF, newdata = as.matrix(kaggledata[-1]))

#Export
# outputDF <- data.frame(PassengerID =numeric(),
#                        Survived=integer(), 
#                        stringsAsFactors=FALSE)
# 
# outputDF$PassengerID <- kaggledata$PassengerId
# outputDF$Survived <- y_pred_kaggle
test <-table(kaggledata$PassengerId,y_pred_kaggle)
write.csv(test,file = "titanic_output3.csv", sep = ",")
