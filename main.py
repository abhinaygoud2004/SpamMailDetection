#Importing the Dependencies
import numpy as np;
import pandas as pd;
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the environment variable
mail_data_path = os.getenv("MAIL_DATA_PATH")
#Data collection & pre-processing
#loading data from csv file to pandas dataframe
raw_mail_data = pd.read_csv(mail_data_path, encoding='latin1')
# print(raw_mail_data.head)

#replace null values with a null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'') 

#checking no.of rows and columns in the dataframe
# print(mail_data.shape)

#label spam mail as 0, ham mail as 1
mail_data.loc[mail_data['v1']=='spam','v1',]=0
mail_data.loc[mail_data['v1']=='ham','v1',]=1

#separating the data as texts and label
X=mail_data['v2']
Y=mail_data['v1']
# print(Y)

#splitting the data into training data and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
# print(X.shape,X_train.shape)

#Feature Extraction
#transform text data to feature vectors that can be used as input to the logistic regression
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

#convert ytrain and ytest values as integers
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')
# print(X_train_features)

#Logistic Regression
model=LogisticRegression()

#training the Logistic Regression model with the training daata
model.fit(X_train_features,Y_train)

#Evaluating the trained model
#prediction on training data

prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train, prediction_on_training_data)
# print(accuracy_on_training_data)

prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test, prediction_on_test_data)
# print(accuracy_on_test_data)

input_mail=["I hope this email finds you well. We have a team meeting scheduled for tomorrow at 10 AM in the conference room. Please come prepared with updates on your ongoing projects."]

#convert text to feature vectors
input_data_features=feature_extraction.transform(input_mail)

#making prediction
prediction=model.predict(input_data_features)

if prediction[0]==1:
    print("Ham mail")
else:
    print("Spam mail")