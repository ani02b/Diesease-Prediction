import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

import pickle

heart_df= pd.read_csv('heart.csv')
print(heart_df.head())

heart_df = heart_df.drop(["chol", "fbs","sex"], axis=1)

x= heart_df.drop(columns= 'target',axis=1)
y= heart_df['target']

scaler =  StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x = standardized_data
print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,stratify=y,random_state=4)

classifier = GaussianNB()
classifier.fit(x_train,y_train)

pickle.dump(classifier,open("heart.pkl","wb"))
