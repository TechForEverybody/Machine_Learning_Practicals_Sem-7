import pandas
import numpy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import seaborn
import matplotlib.pyplot as pyplot

data=pandas.read_csv('./Stress-Lysis.csv')
pyplot.plot(data['Stress Level'],data[['Humidity','Step count','Temperature']])
pyplot.show()
seaborn.pairplot(data.iloc[:,:4], hue="Stress Level",palette="bright")
pyplot.show()
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data['Humidity'])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data['Step count'])
pyplot.scatter(x=[i for i in range(1,len(data)+1)],y=data['Temperature'])
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

DBSCAN_Model=DBSCAN(eps=0.1,min_samples=10)
DBSCAN_Model.fit(Component_Data[:,:2])
Transformed_data=pandas.DataFrame({
    "column1_by_PCA":Component_Data[:,0],
    "column2_by_PCA":Component_Data[:,1],
    "predicted_cluster_by_DBSCAN":DBSCAN_Model.labels_,
})
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==1].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==1].iloc[:,1],c='blue')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==0].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==0].iloc[:,1],c='pink')
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==2].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==2].iloc[:,1],c="black")
pyplot.scatter(x=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==-1].iloc[:,0],y=Transformed_data[Transformed_data['predicted_cluster_by_DBSCAN']==-1].iloc[:,1],c='gray')
pyplot.show()
seaborn.pairplot(Transformed_data.iloc[:,:3], hue="predicted_cluster_by_DBSCAN",palette="bright")
pyplot.show()
