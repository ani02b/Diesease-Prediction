import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree

import pickle

lung_df= pd.read_csv('C:\\Users\\KIIT\\Programs\\Projects\\Content\\lung_cancer.csv')
print(lung_df.head())

lung_df= lung_df.drop(["Name","Surname"],axis=1)

x=lung_df.drop(columns='Result',axis=1)
y=lung_df['Result']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)

pickle.dump(classifier,open("lungs.pkl","wb"))