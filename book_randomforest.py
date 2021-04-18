#you can download the dataset at https://archive.ics.uci.edu/ml/datasets/Car+Evaluation.

#you need to treat each value in the dataset as a string.We consider six attributes
#in the dataset.Here are the attributes along with the possible values they can take:
'''
buying:These will be vhigh,high,med,and low
maint:These will be vhigh,high,med,and low
doors:These will be 2,3,4,5,and more
persons:These will be 2,4,more
lug_boot:These will be small,med,and big
safety:These will be low,med,and high
'''
#given that each line contains strings,we need to assume that all the features are 
#strings and design a classifier.In the previous chapter,we used random forest to
#build a regressor.In this recipe,we will use random forests as a classifier

#We will use car.py file that we already provided to you as reference.Let's go
#ahead and import a couple of packages:
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
#let's load the dataset    
input_file='car.data'
x=[]
count=0
with open(input_file,'r') as f:
    for line in f.readlines():
        data=line[:-1].split(',')
        x.append(data)
x=np.array(x)
#each line contains a comma_separated list of words.Therefore,we parse the input
#file,split each line,and then append the list to the main data.We ignore the
#last character on each line bacause it's a newline character.

#The python packages only work with numerical data,so we need to transform 
#these attributes into something that those packages will understand

#in the previosu chapter,we discussed label encoding.That is what we will use here 
#to convert strings to numbers:
    
#Convert string data to numerical data
label_encoder=[]
x_encoded=np.empty(x.shape)#这个好用！
for i,item in enumerate(x[0]):#逐列操作
    label_encoder.append(preprocessing.LabelEncoder())
    x_encoded[:,i]=label_encoder[-1].fit_transform(x[:,i])
    
x=x_encoded[:,:-1].astype(int)
y=x_encoded[:,-1].astype(int)
#let's train the classifier:
#build a random forest classifier
params={'n_estimators':200,'max_depth':8,'random_state':7}
classifier=RandomForestClassifier(**params)
classifier.fit(x,y)
#you can play around with the n_estimators and max_depth parameters to see
#how they affect the classification accuracy.We wil actually do this soon in a 
#standardized way.

#let's perform cross-validation:
#Cross validation
from sklearn import model_selection #cross_validation 改成model_selection了
accuracy = model_selection.cross_val_score(classifier,x,y,scoring='accuracy',cv=3)
print("Accuracy of the classifier:"+str(round(100*accuracy.mean(),2))+"%")
#once we train the classifier,we need to see how it performs.We use three-fold
#cross-validation to calculate the accuracy here.

#one of the main goals of building a classifier is to use it on isolated and
#unknown data instances.Let's use a single datapoint and see how we can use this 
#classifier to categorize it

'''下面这一段不能运行'''

#Testing encoding on single data instance
input_data=['vhigh','vhigh','2','2','small','low']
input_data_encoded=[-1] * len(input_data)
for i,item in enumerate(input_data):
    input_data_encoded[i]=int(label_encoder[i].transform(np.array([input_data[i]])))

input_data_encoded=np.array(input_data_encoded)

#the first step was to convert that data into numerical data

#we are now ready to predict the output class for this datapoint:

#predict and print output for a particular datapoint:
output_class = classifier.predict(input_data_encoded.reshape(1,-1))
print("Output class",label_encoder[-1].inverse_transform(output_class)[0])

#We ues the predict method to estimate the output class.If we output the encoded
#output label,it wouldn't mean anything to us. Therefore, we use the inverse_transform
#method to convert this label back to its origincal form and print out the output class

from sklearn.model_selection import validation_curve          
# ModuleNotFoundError: No module named 'sklearn.learning_curve'                                                             
classifier=RandomForestClassifier(max_depth=4,random_state=7)
parameter_grid=np.linspace(25,200,8).astype(int)
tr_scores,validation_scores = validation_curve(classifier,x,y,
                                               "n_estimators",parameter_grid,cv=5)
#plot the curve 
import matplotlib.pyplot as plt
plt.figure()
plt.plot(parameter_grid,100*np.average(tr_scores,axis=1))
plt.title('Training curve')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

#let's do the same for the max_depth parameter:
classifier2=RandomForestClassifier(n_estimators=20,random_state=7)
parameter_grid2=np.linspace(2, 10,5).astype(int)
tr_scores2,valid_scores2=validation_curve(classifier2, x, y,"max_depth",parameter_grid2,cv=5)

#let's plot it:
plt.figure()
plt.plot(parameter_grid2,100*np.average(tr_scores2,axis=1),color='black')
plt.title('Validation curve')
plt.xlabel('Maximum depth of the tree')
plt.ylabel('Accuracy')
plt.show()


#extracting learning curves
from sklearn.model_selection import learning_curve
classifier3=RandomForestClassifier(random_state=7)
parameter_grid3=np.array([200,500,800,1100])
tr_sizes3,tr_scores3,valid_scores3=learning_curve(classifier3,x,y,train_sizes=parameter_grid3
                                                  ,cv=5)
#let's plot it
plt.figure()
plt.plot(parameter_grid3,100*np.average(tr_scores3,axis=1),color='black')
plt.title('learning curve')
plt.xlabel('number fo training samples')
plt.ylabel('Accuracy')
plt.show()
#算法改进了 全是100
#ValueError: train_sizes has been interpreted as absolute numbers of training
# samples and must be within (0, 1382],
'''
classifier4=RandomForestClassifier(random_state=7)
parameter_grid4=np.array([2000,5000,8000,11000])
tr_sizes4,tr_scores4,valid_scores4=learning_curve(classifier4,x,y,train_sizes=parameter_grid4
                                                  ,cv=5)                    
                                  
plt.figure()
plt.plot(parameter_grid4,100*np.average(tr_scores4,axis=1),color='black')
plt.title('learning curve')
plt.xlabel('number fo training samples')
plt.ylabel('Accuracy')
plt.show()
'''




