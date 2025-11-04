## Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
Import the packages.

Analyse the data.

Use modelselection and Countvectorizer to preditct the values.

Find the accuracy and display the result.

## Program:

Program to implement the SVM For Spam Mail Detection..
Developed by: yashwanth asv 
RegisterNumber: 212224230309

```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train
x_test
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print (classification_report1)
```

## Output:

<img width="1052" height="62" alt="507404828-dccac94e-94a4-4022-83db-9b23080d9c2f" src="https://github.com/user-attachments/assets/a42b388a-1818-4e43-9fed-8cfb7292f3de" />

DATA

<img width="897" height="257" alt="507404974-1b20199f-6da2-424f-8c7e-995aefb8559c" src="https://github.com/user-attachments/assets/04794a6d-ff00-4b94-9677-c6df205e3dc4" />
<img width="687" height="335" alt="507405096-025cbd52-8ca9-4f8f-884c-7cbd5f1158cc" src="https://github.com/user-attachments/assets/d27d3e0e-66ae-4a9d-8554-9a9f16199033" />

X TRAIN

<img width="302" height="176" alt="507405441-5264cca3-d2a8-46f5-85b0-cb5541b01bdb" src="https://github.com/user-attachments/assets/0e144b9d-a5a3-4734-8339-b59b178356b6" />

X TEST

<img width="1552" height="262" alt="507405612-86142af6-772c-47d1-9df6-79c8751ed390" src="https://github.com/user-attachments/assets/39387284-8eaa-4b57-a5e6-9eae957e3af5" />
<img width="1552" height="307" alt="507405718-80689ac6-0c39-4184-967d-e000e17a2c54" src="https://github.com/user-attachments/assets/a4edea33-ab32-4a4d-9e3e-56f2c4794ed0" />
<img width="886" height="81" alt="507405800-d5640af4-e0d9-4a69-8d3e-299e49feb95b" src="https://github.com/user-attachments/assets/204e5320-62a5-48e9-a2b5-28cd088ad707" />
<img width="857" height="47" alt="507405888-8fc46735-3d02-4f0c-8d89-51c6315672cf" src="https://github.com/user-attachments/assets/2d920ff6-2e5b-4cdb-a81b-12becc8f1fef" />



ACCURACY

<img width="292" height="38" alt="507406017-e2542121-5b9e-472b-82c3-27cbe07debae" src="https://github.com/user-attachments/assets/22d4e498-c4df-4cf0-821b-02b29b40aca0" />

CONFUSION MATRIX

<img width="449" height="73" alt="507406143-9d149c26-27f8-48a2-8c0a-a2c9f7e2b0cd" src="https://github.com/user-attachments/assets/5abe56b8-5924-4d78-b7e3-197b5f10efdb" />


<img width="692" height="300" alt="507409256-aa032844-42dc-437b-a7af-7bd60908a421" src="https://github.com/user-attachments/assets/71bc9e58-7c12-477d-a1cc-d355c76cdbc1" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
