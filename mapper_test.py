# -*- coding: utf-8 -*-

# Import the class
import kmapper as km
from sklearn import cluster, preprocessing, manifold, decomposition
import processing.fifa as fifa
from matplotlib import pyplot

# 1 - get data
# Some sample data
from sklearn import datasets
#data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.1)

data = fifa.get_processed_data()

# Initialize
mapper = km.KeplerMapper(verbose=1)

# distance_matrix
print(data)
# filtering with distance_matrix
# Fit to and transform the data -> 여기서 normalize도 함.
projected_data = mapper.fit_transform(
    data,
    projection=None,
    scaler=preprocessing.MinMaxScaler(),
    #scaler=preprocessing.MinMaxScaler(),
    ) # X-Y axis



pca = decomposition.PCA(n_components=2)
pca_data = pca.fit_transform(projected_data)

pyplot.scatter(pca_data[:,0],pca_data[:,1],s=1)
pyplot.show()

#print(pca_data.sum())

# projected_data에 square distance matrix 가  나올 수 있음
# 지금은 그냥 scaling만 한거...
# map에서 square dist matrix가 드가면 뭘 따로 처리하나...?

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(pca_data, data, nr_cubes=10, precomputed=False)

# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")

#

