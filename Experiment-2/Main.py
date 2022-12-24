import pandas
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
import seaborn
import matplotlib.pyplot as pyplot
# this are the output classes, only two classes are there paitient will dead or alive
classes_list=['Alive','Dead']
data=pandas.read_csv('./shuffled_heart_failure_clinical_records_dataset.csv')
print(data.describe())

Training_Data=data[:250]
Testing_Data=data[250:]

training_Labels=Training_Data['DEATH_EVENT']
training_Features=Training_Data.drop(['DEATH_EVENT'],axis=1)
testing_Labels=Testing_Data['DEATH_EVENT']
testing_Features=Testing_Data.drop(['DEATH_EVENT'],axis=1)

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

print("""
testing on new Data

for,
age=45 (age is 45)
anaemia=0 (no anaemia symptom)
creatinine_phosphokinase=552 (level of creatinine phosphokinase in heart)
diabetes=0 
ejection_fraction=30 (ejection fraction reading is 20 for heart)
high_blood_pressure=0 (no presence of high blood pressure)
platelets=265000 (platelets level in heart)
serum_creatinine=1.9 (serum creatinine value)
serum_sodium=130 (
sex=1 (man)
smoking=0 (no smoking habbit)
time=4 (Condition Follow-up period in days)
""")
new_data=[45,0,552,0,50,0,265000,1.9,130,1,0,4]
print("Patient will be ",classes_list[model.predict([new_data])[0]])
