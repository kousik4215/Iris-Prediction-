
# A very simple Flask Hello World app for you to get started with...

from flask import Flask

app = Flask(__name__)


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os
os.chdir('/home/kousik/mit workshop')
data=pd.read_csv('iris.csv')
from sklearn.model_selection import train_test_split
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
import warnings
warnings.filterwarnings('ignore')
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)
test=logmodel.predict([[5.9,3.0,5.1,1.8]])
from flask import Flask,request,jsonify
app= Flask('Hi friends')
@app.route('/hello')
def new():
    return "arey tharun project chey raaa"
@app.route('/<float:sepal_length>/<float:sepal_width>/<float:petal_length>/<float:petal_width>')
def test(sepal_length,sepal_width,petal_length,petal_width):
    p=[]
    p +=[sepal_length,sepal_width,petal_length,petal_width]
         
    arr=np.array([p])
    predict=logmodel.predict(arr)
    
    if predict == "Iris-virginica":
        result = {'result':'Iris-virginica'}
    elif predict== "Iris-setosa":
        result = {'result':"Iris-setosa"}
    else:
         result = {'result':"Iris-versicolor"}
        
        
    return result
app.run()
