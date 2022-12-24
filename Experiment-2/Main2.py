import pandas
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
import seaborn
import matplotlib.pyplot as pyplot

# this are the output classes, only two classes are there paitient will have stroke or not
classes_list=['Not_Failed','Failed']
data=pandas.read_csv('./Brain_failure_healthcare-dataset-stroke-data.csv')

data=data.replace("Male",1)
data=data.replace("Female",0)
data=data.replace("Other",0)
data=data.drop(['ever_married','work_type','Residence_type'],axis=1)
data=data.replace("never smoked",0)
data=data.replace("Unknown",0)
data=data.replace("formerly smoked",1)
data=data.replace("smokes",1)
data['bmi']=data['bmi'].fillna(value=data['bmi'].mean())
data['bmi']=data['bmi'].fillna(value=data['bmi'].mean())

shuffled_Data=data.sample(frac=1)
failed_data=data[data['stroke']==1]
not_failed_data=data[data['stroke']==0]
not_failed_data=not_failed_data.head(249)
Analysis_Data=pandas.concat([failed_data,not_failed_data],axis=0)
Analysis_Data=Analysis_Data.sample(frac=1)
Training_Data = Analysis_Data[:450]
Testing_Data = Analysis_Data[450:]
training_Labels=Training_Data['stroke']
training_Features=Training_Data.drop(['stroke'],axis=1)
testing_Labels=Testing_Data['stroke']
testing_Features=Testing_Data.drop(['stroke'],axis=1)

model=LogisticRegression(max_iter=1000)
model.fit(training_Features,training_Labels)

predicted_values=model.predict(testing_Features)
predicted_values=list(predicted_values)
testing_Labels=list(testing_Labels)
print("Actual Values --> Predicted values")
for i in range(len(predicted_values)):
    print("          ",testing_Labels[i]," --> ",predicted_values[i])
classification_data=confusion_matrix(testing_Labels,predicted_values)
seaborn.heatmap(classification_data,annot=True,xticklabels=classes_list,yticklabels=classes_list)
pyplot.show()
print(classification_report(testing_Labels,predicted_values,target_names=classes_list))


new_data=[123312,1,21,0,0,150,19,0]
print(classes_list[model.predict([new_data])[0]])
