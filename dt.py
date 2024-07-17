import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import  classification_report,confusion_matrix
from sklearn.ensemble import  RandomForestClassifier

data=pd.read_csv("kyphosis.csv")
df=pd.DataFrame(data)
# print(df.info())
# sns.pairplot(df)
# plt.show()
X=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']
# x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=155)

#Decision Tree Algorithm

# dtree=DecisionTreeClassifier()
# dtree.fit(x_train,y_train)
# pred=dtree.predict(x_test)
# # print(pred)
#
# print(confusion_matrix(y_test,pred))
# print(classification_report(y_test,pred))

# Random forest Algorithm
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=190)
dtree=RandomForestClassifier(n_estimators=200)
dtree.fit(x_train,y_train)
pred=dtree.predict(x_test)
# print(pred)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))