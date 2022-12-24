import pandas
import numpy
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.decomposition import PCA
import seaborn
import matplotlib.pyplot as pyplot
import warnings
warnings.filterwarnings("ignore")
model_for_normal_data=svm.SVC()
model_for_Transformed_data=svm.SVC()
temp_model=svm.SVC()
classes_list=["rice", "maize", "cotton", "coconut",  "orange", "apple", "watermelon","jute", "mango","coffee"]
def get_class_number(class_name):
    if class_name=='rice':
        return 0
    if class_name=='maize':
        return 1
    if class_name=='cotton':
        return 2
    if class_name=='coconut':
        return 3
    if class_name=='orange':
        return 4
    if class_name=='apple':
        return 5
    if class_name=='jute':
        return 6
    if class_name=='mango':
        return 7
    if class_name=='watermelon':
        return 8
    if class_name=='coffee':
        return 9
data=pandas.read_csv('./Crop_Suggestion.csv')
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data['temperature'],color = 'gray')
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data['humidity'],color = 'red')
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data['rainfall'],color = 'yellow')
pyplot.show()
seaborn.pairplot(data.iloc[:,:4], hue="label",palette="bright")
pyplot.show()

Principal_Component_Analyser=PCA(n_components=2)
Component_Data=Principal_Component_Analyser.fit_transform(data.iloc[:,:3])
Transformed_Data=pandas.DataFrame({"label":data['label']})
Transformed_Data['column1']=[i[0] for i in Component_Data]
Transformed_Data['column2']=[i[1] for i in Component_Data]
seaborn.pairplot(Transformed_Data,hue="label",palette="bright")
pyplot.show()
pyplot.scatter(x=[i for i in range(1,len(Transformed_Data)+1)],y=Transformed_Data['column1'],color = 'coral')
pyplot.scatter(x=[i for i in range(1,len(Transformed_Data)+1)],y=Transformed_Data['column2'],color='blue')
pyplot.show()

shuffled_Data=data.sample(frac=1)
Training_Data = shuffled_Data[:850]
Testing_Data = shuffled_Data[850:]
training_Labels=Training_Data['label']
training_Features=Training_Data.drop(['label'],axis=1)
testing_Labels=Testing_Data['label']
testing_Features=Testing_Data.drop(['label'],axis=1)
model_for_normal_data.fit(training_Features,training_Labels)
training_predicted_values=model_for_normal_data.predict(training_Features)
training_classification_data=confusion_matrix(training_Labels,training_predicted_values)
seaborn.heatmap(training_classification_data,annot=True,xticklabels=classes_list,yticklabels=classes_list)
print(classification_report(training_Labels,training_predicted_values,target_names=classes_list))
pyplot.show()
predicted_values=model_for_normal_data.predict(testing_Features)
classification_data=confusion_matrix(testing_Labels,predicted_values)
seaborn.heatmap(classification_data,annot=True,xticklabels=classes_list,yticklabels=classes_list)
print(classification_report(testing_Labels,predicted_values,target_names=classes_list))
pyplot.show()

Tranformed_shuffled_Data=Transformed_Data.sample(frac=1)
Tranformed_Training_Data = Tranformed_shuffled_Data[:850]
Tranformed_Testing_Data = Tranformed_shuffled_Data[850:]
Tranformed_training_Labels=Tranformed_Training_Data['label']
Tranformed_training_Features=Tranformed_Training_Data.drop(['label'],axis=1)
Tranformed_testing_Labels=Tranformed_Testing_Data['label']
Tranformed_testing_Features=Tranformed_Testing_Data.drop(['label'],axis=1)
model_for_Transformed_data.fit(Tranformed_training_Features,Tranformed_training_Labels)
Transformed_training_predicted_values=model_for_Transformed_data.predict(Tranformed_training_Features)
Transformed_training_classification_data=confusion_matrix(Tranformed_training_Labels,Transformed_training_predicted_values)
seaborn.heatmap(Transformed_training_classification_data,annot=True,xticklabels=classes_list,yticklabels=classes_list)
pyplot.show()
print(classification_report(Tranformed_training_Labels,Transformed_training_predicted_values,target_names=classes_list))
Transformed_predicted_values=model_for_Transformed_data.predict(Tranformed_testing_Features)
Transformed_classification_data=confusion_matrix(Tranformed_testing_Labels,Transformed_predicted_values)
seaborn.heatmap(Transformed_classification_data,annot=True,xticklabels=classes_list,yticklabels=classes_list)
pyplot.show()
print(classification_report(Tranformed_testing_Labels,Transformed_predicted_values,target_names=classes_list))

temp_shuffled_Data=data.sample(frac=1)
temp_Training_Data = temp_shuffled_Data[:850]
temp_Testing_Data = temp_shuffled_Data[850:]
temp_training_Labels=temp_Training_Data['label']
temp_training_Features=temp_Training_Data.drop(['label','rainfall'],axis=1)
temp_testing_Labels=temp_Testing_Data['label']
temp_testing_Features=temp_Testing_Data.drop(['label','rainfall'],axis=1)
temp_model.fit(temp_training_Features,temp_training_Labels)
temp_predicted_values=temp_model.predict(temp_testing_Features)
temp_classification_data=confusion_matrix(temp_testing_Labels,temp_predicted_values)
seaborn.heatmap(temp_classification_data,annot=True,xticklabels=classes_list,yticklabels=classes_list)
pyplot.show()
print(classification_report(temp_testing_Labels,temp_predicted_values,target_names=classes_list))
