import pandas
import numpy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import seaborn
import matplotlib.pyplot as pyplot

data=pandas.read_csv('./Crop_Suggestion.csv')
pyplot.plot([i for i in range(1,len(data)+1)],data[['temperature','humidity','rainfall']])
pyplot.show()
seaborn.pairplot(data.iloc[:,:4], hue="label",palette="bright")
pyplot.show()
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data['rainfall'])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data['humidity'])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data['temperature'])
pyplot.show()

standard_scaler = StandardScaler()
data_scaled=standard_scaler.fit_transform(data.iloc[:,:3])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data_scaled[:,0])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data_scaled[:,1])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data_scaled[:,2])
pyplot.show()
data_normalized = normalize(data_scaled)
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data_normalized[:,0])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data_normalized[:,1])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data_normalized[:,2])
pyplot.show()
Principal_Component_Analyser=PCA(n_components=2)
Component_Data=Principal_Component_Analyser.fit_transform(data_normalized[:,:3])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=Component_Data[:,0])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=Component_Data[:,1])
pyplot.show()

DBSCAN_Model=DBSCAN(eps=0.08,min_samples=10)
DBSCAN_Model.fit(Component_Data[:,:2])
Transformed_data=pandas.DataFrame({
    "column1_by_PCA":Component_Data[:,0],
    "column2_by_PCA":Component_Data[:,1],
    "predicted_cluster_by_DBSCAN":DBSCAN_Model.labels_,
})
Transformed_data.iloc[:,:2].plot()
pyplot.show()
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==1].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==1].iloc[:,1],c='blue')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==0].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==0].iloc[:,1],c='pink')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==2].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==2].iloc[:,1],c="black")
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==3].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==3].iloc[:,1],c='green')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==4].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==4].iloc[:,1],c='red')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==5].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==5].iloc[:,1],c='aqua')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==6].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==6].iloc[:,1],c='coral')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==7].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==7].iloc[:,1],c='aquamarine')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==8].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==8].iloc[:,1],c='royalblue')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==9].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==9].iloc[:,1],c='yellow')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==-1].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==-1].iloc[:,1],c='gray')
pyplot.show()
seaborn.pairplot(Transformed_data.iloc[:,:3], hue="predicted_cluster_by_DBSCAN",palette="bright")
pyplot.show()
