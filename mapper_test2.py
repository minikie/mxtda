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
samples = 3000
cubes = 30
over_lap = 0.15
data = fifa.get_processed_data(sample=samples)

# Initialize
mapper = km.KeplerMapper(verbose=2)

# distance_matrix
# filtering with distance_matrix
# Fit to and transform the data -> 여기서 normalize도 함.
projected_data = mapper.fit_transform(
    data,
    projection=None,
    scaler=preprocessing.Normalizer(),
    #scaler=preprocessing.MinMaxScaler(),
    ) # X-Y axis


# pca = decomposition.PCA(n_components=2)
# pca_data = pca.fit_transform(projected_data)
#
# pyplot.scatter(pca_data[:,0],pca_data[:,1],s=1)
# pyplot.show()

#print(pca_data.sum())

# projected_data에 square distance matrix 가  나올 수 있음
# 지금은 그냥 scaling만 한거...
# map에서 square dist matrix가 드가면 뭘 따로 처리하나...?

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data[:,1],
                   projected_data,
                   cover=km.Cover(n_cubes=cubes, perc_overlap=over_lap),
                   clusterer=cluster.DBSCAN(eps=0.2, min_samples=3),
                   precomputed=False)

# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="make_circles(n_samples={0}, cube={1}, overlap={2})".format(samples, cubes, over_lap))

