import pandas
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report
import seaborn
import matplotlib.pyplot as pyplot

data=pandas.read_csv('./Crop_recommendation.csv')
print(data.columns)
print(data)
print(data.info())
print(data.describe())
print(data.corr())
pyplot.plot(data['label'],data[['N','P','K','temperature','humidity','rainfall','ph']])
pyplot.show()
print(data['label'].value_counts())

data=data.drop(['N','P','K','ph'],axis=1)
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
Analysis_data=data[data['label']=='rice']
for i in range(1,len(classes_list)):
    Analysis_data=pandas.concat([Analysis_data,data[data['label']==classes_list[i]]])
data=Analysis_data
print(data['label'].value_counts())
data['class_number']=data['label'].apply(get_class_number)

seaborn.pairplot(data.iloc[:,:4], hue="label",palette="bright")
pyplot.show()
pyplot.plot([i for i in range(1,len(data)+1)],data[['temperature','humidity','rainfall']])
pyplot.show()
plotter = pyplot.subplot(projection='3d')
plotter.scatter3D(data.to_numpy()[:,0], data.to_numpy()[:, 1] ,data.to_numpy()[:, 2],c=data['class_number'], cmap='autumn')
plotter.set_xlabel('temperature')
plotter.set_ylabel('humidity')
plotter.set_zlabel('rainfall')
pyplot.show()

shuffled_Data=data.sample(frac=1)
Training_Data = shuffled_Data[:850]
Testing_Data = shuffled_Data[850:]
training_Labels=Training_Data['label']
training_class_numbers=Training_Data['class_number']
training_Features=Training_Data.drop(['label','class_number'],axis=1)
testing_Labels=Testing_Data['label']
testing_class_numbers=Testing_Data['class_number']
testing_Features=Testing_Data.drop(['label','class_number'],axis=1)

model=svm.SVC(kernel='linear')
model.fit(training_Features,training_Labels)
print(model.support_vectors_)

pyplot.scatter(training_Features.to_numpy()[:, 0],training_Features.to_numpy()[:, 1],training_Features.to_numpy()[:, 2],c=training_class_numbers, cmap='autumn')
pyplot.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],model.support_vectors_[:,2])
pyplot.show()
plotter = pyplot.subplot(projection='3d')
plotter.scatter3D(training_Features.to_numpy()[:, 0], training_Features.to_numpy()[:, 1] ,training_Features.to_numpy()[:, 2],c=training_class_numbers, cmap='autumn')
plotter.plot3D(model.support_vectors_[:,0],model.support_vectors_[:,1],model.support_vectors_[:,2])
plotter.set_xlabel('temperature')
plotter.set_ylabel('humidity')
plotter.set_zlabel('rainfall')
pyplot.show()
training_predicted_values=model.predict(training_Features)
training_classification_data=confusion_matrix(training_Labels,training_predicted_values)
seaborn.heatmap(training_classification_data,annot=True,xticklabels=classes_list,yticklabels=classes_list)
print(classification_report(training_Labels,training_predicted_values,target_names=classes_list))
pyplot.show()

predicted_values=model.predict(testing_Features)
predicted_values=list(predicted_values)
testing_Labels=list(testing_Labels)
print("Actual Values --> Predicted values")
for i in range(len(predicted_values)):
    print("     ",testing_Labels[i]," --> ",predicted_values[i])
classification_data=confusion_matrix(testing_Labels,predicted_values)
seaborn.heatmap(classification_data,annot=True,xticklabels=classes_list,yticklabels=classes_list)
pyplot.show()
print(classification_report(testing_Labels,predicted_values,target_names=classes_list))

new_data=[30,49,95]
print("Predicted Value is -> ",model.predict([new_data])[0])