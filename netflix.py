import numpy as np
import pandas as pd
import matplotlib.pyplot as p
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
data=pd.read_csv("D:/netflix_titles.csv")
n=data.shape;
# sh=data.shape
# n1=sh[0]
# print(n1)
# data.info() 
while(True):
    a=int(input("Choose an option\n 1.to search a title in dataset\n2.Different rating are in  Netflix\n3.Maximum ratings used\n4.Search a category in movie/tvshow\n5.Exit\n"))
    # to search a titl
    if a==1:
        n=input("Enter the movie name:")
        a=data[data["title"].isin([n])]
        print(a)
    # shows are released in a particular year
    elif a==2:
        a=data["rating"].unique()
        b=data["rating"].nunique()
        print(a)
        print(b)     
    # different types of rating are in Netflix
    elif a==3:
        x=['66 min ', '74 min ' ,'84 min' ,'G' ,'NC-17', 'NR', 'PG', 'PG-13' ,'R' ,'TV-14',
         'TV-G', 'TV-MA', 'TV-PG' ,'TV-Y' ,'TV-Y7','TV-Y7-FV', 'UR']
        a=data.groupby("rating").type.count()
        p.xticks(rotation=90)
        p.bar(x,data.groupby("rating").type.count())
        p.show()
    # search a category in movie/tvshow
    #
    elif a==4:
        n=input("Enter the type movie or tv show")
        m=input("Enter  the category")
        a=data[(data["type"]==n)&(data["listed_in"]==m)]
        print(a)
       # to exit
    elif a==5:
        n=n[0];
        number  = preprocessing.LabelEncoder()
        data['rating'] = number.fit_transform(data.rating)
        data['type'] = number.fit_transform(data.type)
        data['director'] = number.fit_transform(data.director)
        data['country'] = number.fit_transform(data.country)
        data['listed_in'] = number.fit_transform(data.listed_in)
        d=data[["type","director","country","listed_in"]]
        X=d
        y=data[["rating"]]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
        print("\nthe training set\n")
        print(X_train)
        a=data.dtypes
        print("\nthe training set\n")
        print(y_train)
        print(y_train.shape)
        print("\nthe test set\n")
        print(X_test)
        print(X_test.shape)
        knn=KNeighborsClassifier(n_neighbors=7,metric='euclidean')
        print(knn.fit(X_train,y_train))
        print(knn)1
        y_predict1=knn.predict(X_test)
        print(y_predict1)
        print(y_test.value_counts())
        cm=confusion_matrix(y_test,y_predict1)
        print(cm)
        print(accuracy_score(y_test,y_predict1))
        df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        p.figure(figsize = (10,7))
        sns.set(font_scale=1.4)#for label size
        sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 5})
  elif a==6:
         break
}
