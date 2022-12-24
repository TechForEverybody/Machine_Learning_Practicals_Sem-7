import pandas
import numpy
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report
import seaborn
import matplotlib.pyplot as pyplot
from mpl_toolkits import mplot3d
pyplot.rcParams["figure.figsize"] = (15, 12)
classes_list=['Low Stress','Normal Stress','High Stress']
data=pandas.read_csv('./Stress-Lysis.csv')

print(data)
print(data.describe())
print(data['Stress Level'].value_counts())
seaborn.pairplot(data, hue="Stress Level",palette="bright")
pyplot.show()

plotter = pyplot.subplot(projection='3d')
plotter.scatter3D(data.to_numpy()[:, 0], data.to_numpy()[:, 1] ,data.to_numpy()[:, 2],c=data['Stress Level'], cmap='gray')
plotter.set_xlabel('Humidity')
plotter.set_ylabel('Temperature')
plotter.set_zlabel('Step count')

shuffled_Data=data.sample(frac=1)

Training_Data = shuffled_Data[:1600]
Testing_Data = shuffled_Data[1600:]
training_Labels=Training_Data['Stress Level']
training_Features=Training_Data.drop(['Stress Level'],axis=1)
print(training_Features.info())
print(training_Labels.value_counts())
testing_Labels=Testing_Data['Stress Level']
testing_Features=Testing_Data.drop(['Stress Level'],axis=1)
print(testing_Features.info())
print(testing_Labels.value_counts())

model=svm.SVC()
model.fit(training_Features,training_Labels)
print(model.support_vectors_)
print(model.n_support_)

plotter = pyplot.subplot(projection='3d')
plotter.scatter3D(training_Features.to_numpy()[:, 0], training_Features.to_numpy()[:, 1] ,training_Features.to_numpy()[:, 2],c=training_Labels, cmap='autumn')
plotter.plot3D(model.support_vectors_[:,0],model.support_vectors_[:,1],model.support_vectors_[:,2])
plotter.set_xlabel('Humidity')
plotter.set_ylabel('Temperature')
plotter.set_zlabel('Step count')
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
    print("          ",testing_Labels[i]," --> ",predicted_values[i])
classification_data=confusion_matrix(testing_Labels,predicted_values)
seaborn.heatmap(classification_data,annot=True,xticklabels=classes_list,yticklabels=classes_list)
pyplot.show()

print(classification_report(testing_Labels,predicted_values,target_names=classes_list))
new_data=[20,95,200]
print(classes_list[model.predict([new_data])[0]])