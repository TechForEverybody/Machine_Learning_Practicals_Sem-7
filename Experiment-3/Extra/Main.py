import pandas
import numpy
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,classification_report
import seaborn
import matplotlib.pyplot as pyplot


# this are the output classes, only two classes are there paitient will dead or alive
classes_list=['Not_Failed','Failed']

data=pandas.read_csv('./Brain_failure_healthcare-dataset-stroke-data.csv')

data=data.drop(['id','bmi'],axis=1)


data=data.replace("Male",1)
data=data.replace("Female",0)
data=data.replace("Other",0)


data=data.drop(['ever_married','work_type','Residence_type'],axis=1)

data=data.replace("never smoked",0)
data=data.replace("Unknown",0)
data=data.replace("formerly smoked",1)
data=data.replace("smokes",1)

seaborn.pairplot(data.iloc[:,:7], hue="stroke",palette="bright")
pyplot.show()

pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data.iloc[:,0])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data.iloc[:,1])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data.iloc[:,2])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data.iloc[:,3])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data.iloc[:,4])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data.iloc[:,5])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data.iloc[:,6])
pyplot.show()


shuffled_Data=data.sample(frac=1)
training_value=int(len(shuffled_Data)*0.85)

Training_Data = shuffled_Data[:training_value]
Testing_Data = shuffled_Data[training_value:]

training_Labels=Training_Data['stroke']
training_Features=Training_Data.drop(['stroke'],axis=1)

testing_Labels=Testing_Data['stroke']
testing_Features=Testing_Data.drop(['stroke'],axis=1)


svmmodel=svm.LinearSVC()
bagging_classifier = BaggingClassifier(base_estimator=svmmodel)
bagging_classifier.fit(training_Features,training_Labels)


training_predicted_values=bagging_classifier.predict(training_Features)
training_classification_data=confusion_matrix(training_Labels,training_predicted_values)
seaborn.heatmap(training_classification_data,annot=True)
pyplot.show()


print(classification_report(training_Labels,training_predicted_values))
predicted_values=bagging_classifier.predict(testing_Features)
classification_data=confusion_matrix(predicted_values,testing_Labels)
seaborn.heatmap(classification_data,annot=True)
pyplot.show()


print(classification_report(testing_Labels,predicted_values))

XGBoost_classifier = xgboost.XGBClassifier()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
training_Labels = le.fit_transform(training_Labels)
testing_Labels=le.transform(testing_Labels)
XGBoost_classifier.fit(training_Features,training_Labels)

training_predicted_values=XGBoost_classifier.predict(training_Features)
training_classification_data=confusion_matrix(training_Labels,training_predicted_values)
seaborn.heatmap(training_classification_data,annot=True)
pyplot.show()

print(classification_report(training_Labels,training_predicted_values))

predicted_values=XGBoost_classifier.predict(testing_Features)
classification_data=confusion_matrix(predicted_values,testing_Labels)
seaborn.heatmap(classification_data,annot=True)
pyplot.show()

print(classification_report(testing_Labels,predicted_values))

randomForestClassifier=RandomForestClassifier()

randomForestClassifier.fit(training_Features,training_Labels)

training_predicted_values=randomForestClassifier.predict(training_Features)
training_classification_data=confusion_matrix(training_Labels,training_predicted_values)
seaborn.heatmap(training_classification_data,annot=True)
pyplot.show()

print(classification_report(training_Labels,training_predicted_values))

predicted_values=randomForestClassifier.predict(testing_Features)
classification_data=confusion_matrix(predicted_values,testing_Labels)
seaborn.heatmap(classification_data,annot=True)
pyplot.show()

print(classification_report(testing_Labels,predicted_values))

gradientBoostingClassifier=GradientBoostingClassifier()

gradientBoostingClassifier.fit(training_Features,training_Labels)

training_predicted_values=gradientBoostingClassifier.predict(training_Features)
training_classification_data=confusion_matrix(training_Labels,training_predicted_values)
seaborn.heatmap(training_classification_data,annot=True)
pyplot.show()

print(classification_report(training_Labels,training_predicted_values))


predicted_values=gradientBoostingClassifier.predict(testing_Features)
classification_data=confusion_matrix(predicted_values,testing_Labels)
seaborn.heatmap(classification_data,annot=True)
pyplot.show()

print(classification_report(testing_Labels,predicted_values))

new_data=[1,21,0,0,150,0]

print(classes_list[bagging_classifier.predict([new_data])[0]])

print(classes_list[randomForestClassifier.predict([new_data])[0]])
print(classes_list[gradientBoostingClassifier.predict([new_data])[0]])