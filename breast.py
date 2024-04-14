import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import pickle

brst_df = pd.read_csv('C:\\Users\\KIIT\\Programs\\Projects\\Content\\breast.csv')
print(brst_df.head())

brst_df = brst_df.dropna(axis=1)

x= brst_df.drop(columns= 'diagnosis',axis=1)# all fearures
y= brst_df['diagnosis']#labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

pickle.dump(classifier,open("breast.pkl","wb"))