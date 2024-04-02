import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import pickle

kid_df= pd.read_csv('C:\\Users\\KIIT\\Programs\\Projects\\Content\\kidney_disease.csv')
print(kid_df.head())

kid_df = kid_df[kid_df['classification'] != 'ckd\t']

# Reset the index to ensure it's continuous after dropping rows
kid_df.reset_index(drop=True, inplace=True)
kid_df['classification'] = kid_df['classification'].replace({'ckd': 1, 'notckd': 0})

#kid_df['age'].fillna(kid_df['age'].mean(), inplace=True)
#kid_df['sg'].fillna(kid_df['sg'].mean(), inplace=True)

kid_df = kid_df.drop(["bp","sc","pot","hemo","id","pcv","wc","rc","rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane","sod"], axis=1)
x= kid_df.drop(columns= 'classification',axis=1)# all features
y= kid_df['classification']#labels

"""
scaler =  StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x = standardized_data
print(x)
"""

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,stratify=y,random_state=10)

classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)

pickle.dump(classifier,open("kidney.pkl","wb"))