import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import pickle

dia_df= pd.read_csv('C:\\Users\\KIIT\\Programs\\Projects\\Content\\diabetes.csv')
print(dia_df.head())

dia_df = dia_df.drop(["Pregnancies"], axis=1)
dia_df = dia_df.drop(["SkinThickness"], axis=1)

x= dia_df.drop(columns= 'Outcome',axis=1)# all fearures
y= dia_df['Outcome']

scaler =  StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x = standardized_data
print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,stratify=y,random_state=10)

classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)

pickle.dump(classifier,open("diabetes.pkl","wb"))