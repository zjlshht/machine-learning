#data from https://archive.ics.uci.edu/ml/datasets/Census+Income
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import numpy as np
input_file='adult.data'
x=[]
y=[]
count_lessthan50k=0
count_morethan50k=0
num_images_threshold=10000
with open(input_file,'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        
        data = line[:-1].split(', ')
        
        if data[-1]=='<=50K' and count_lessthan50k < num_images_threshold:
            x.append(data)
            count_lessthan50k = count_lessthan50k+1
        elif data[-1]=='>50K' and count_morethan50k < num_images_threshold:
            x.append(data)
            count_morethan50k = count_morethan50k +1
        
        if count_lessthan50k >= num_images_threshold and count_morethan50k>= num_images_threshold:
            break 
x=np.array(x)

#Convert string data to numerical data
label_encoder=[]
x_encoded=np.empty(x.shape)
for i,item in enumerate(x[0]):
    if item.isdigit():
        x_encoded[:,i]=x[:,i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        x_encoded[:,i]=label_encoder[-1].fit_transform(x[:,i])
        
x=x_encoded[:,:-1].astype(int)
y=x_encoded[:,-1].astype(int)

#bulid a classifier 
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(x,y)

#let's split the data into training and testing to extract performance metrics:
#cross validation 
from sklearn import model_selection
x_tr,x_te,y_tr,y_te=model_selection.train_test_split(x, y,test_size=0.25,random_state=5)
classifier_gaussiannb=GaussianNB()
classifier_gaussiannb.fit(x_tr,y_tr)
y_te_pred=classifier_gaussiannb.predict(x_te)

#let's extract performance metrics:
#compute F1 score of the classifier
f1 = model_selection.cross_val_score(classifier_gaussiannb,x,y,scoring='f1_weighted',cv=5)
print("F1 score:"+str(round(100*f1.mean(),2))+"%")

#let's see how to classify a single data instance
input_data=['39','State-gov','77516','Bachelors','13','Never-married','Adm-clerical',
            'Not-in-family','White','Male','2174','0','40','United-States']
count=0
input_data_encoded = [-1]*len(input_data)
for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i]=int(input_data[i])
    else:
        input_data_encoded[i]=int(label_encoder[count].transform(np.array([input_data[i]])))
        #transform 需要一个数组 不能是一个字符串！！！
        count=count+1
    
input_data_encoded=np.array(input_data_encoded)
    
'''
d=enumerate(input_data)

for i,j in d:
    print(i,j)
    
0 39
1 State-gov
2 77516
3 Bachelors
4 13
5 Never-married
6 Adm-clerical
7 Not-in-family
8 White
9 Male
10 2174
11 0
12 40
13 United-States
'''
# we are now ready to classify it:
#Predict and print output for a particular datapoint
output_class=classifier_gaussiannb.predict(input_data_encoded.reshape(-1,1))
print(label_encoder[-1].inverse_transform(output_class)[0])

