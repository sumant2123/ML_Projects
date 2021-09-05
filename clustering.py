from sklearn.cluster import KMeans
import numpy as np
from collections import  defaultdict
import scipy.linalg as la
from sklearn.cluster import DBSCAN

points=[[2,3],[7,9],[11,4],[3,3],[14,12],[4,5],[12,5],[9,7],[6,4],[2,9],[4,7],[6,8]]

# DBSCAN

clustering = DBSCAN(eps=6, min_samples=3).fit(points)
print("DBSCAN Clustering lables fro each label:",clustering.labels_)


# KMeans Code

kmeans = KMeans(n_clusters=3, random_state=1).fit(points)
labels=kmeans.labels_
centroids=kmeans.cluster_centers_
for index,x in enumerate(centroids):
    print("Centroid of Cluster {0} is {1}".format(index+1,x))


def calculateeuclidien(x,y):
    return np.linalg.norm(x-y)

clusters=defaultdict(list)
sses=defaultdict(list)

for index,point in enumerate(points):
    clusters[labels[index]].append(point)
    sses[labels[index]].append(pow(calculateeuclidien(point, centroids[labels[index]]), 2))

for x in sorted(clusters.keys()):
    print("Points in cluster {0} are {1}".format(x,clusters[x]))
for x in sses.keys():
    print("SSE for cluster {0} is {1}".format(x,sum(sses[x])))

# Silhoutte coeff

def getsiiluhetecoeff(reqpoint,reqcluster):
    a=0
    for x in clusters[reqcluster]:
        if x !=reqpoint:
            a+=calculateeuclidien(np.array(reqpoint), x)
    a=a/(len(clusters[reqcluster])-1)
    b=[]
    for x in clusters.keys():
        if x !=reqcluster:
            temp=0
            for y in clusters[x]:
                temp += calculateeuclidien(np.array(reqpoint), y)
            temp=temp/len(clusters[x])
            b.append(temp)
    b=min(b)
    s=(b-a)/max(a,b)
    return s

reqcluster = 0
for index,point in enumerate(points):
    if point == [7,9]:
        reqcluster = labels[index]
for x in clusters[reqcluster]:
    print("Silhouette coeff of {0} is {1}".format(str(x),getsiiluhetecoeff(x,reqcluster)))


# Spectral Clustering (Eigen Vectors and Eigen Values)


temp=np.array([[3,0,0,-1,0,-1,0,0,-1,0,0,0],[0,4,0,0,-1,0,0,-1,0,0,-1,-1],[0,0,3,0,0,0,-1,-1,-1,0,0,0],[-1,0,0,3,0,-1,0,0,-1,0,0,0],[0,-1,0,0,3,0,-1,-1,0,0,0,0],
                 [-1,0,0,-1,0,5,0,0,-1,-1,-1,0],[0,0,-1,0,-1,0,4,-1,-1,0,0,0],[0,-1,-1,0,-1,0,-1,5,0,0,0,-1],[-1,0,-1,-1,0,-1,-1,0,6,0,-1,0],
                 [0,0,0,0,0,-1,0,0,0,3,-1,-1],[0,-1,0,0,0,-1,0,0,-1,-1,5,-1],[0,-1,0,0,0,0,0,-1,0,-1,-1,4]])

evals, evecs = la.eig(temp)
evals = evals.real
print(evals)
print(evecs)
x=evecs[:,0:3]
kmeans = KMeans(n_clusters=3, random_state=1).fit(list(x))
print("Cluster Labels are:",kmeans.labels_)
print("Cluster Centers are:",kmeans.cluster_centers_)
clusters=defaultdict(list)
labels=kmeans.labels_
for index,point in enumerate(points):
    clusters[labels[index]].append(point)

for x in sorted(clusters.keys()):
    print("Points in cluster {0} are {1}".format(x,clusters[x]))



# Fuzzy C means


centers=[[1,2],[8,7],[14,5]]
membership=defaultdict(list)
for x in points:
    for y in centers:
        temp=0
        for z in centers:
            temp+=pow(calculateeuclidien(np.array(x),y)/calculateeuclidien(np.array(x),z),2)
        membership[str(x)[1:-1]].append(1/temp)

print(membership)

newcentroids=defaultdict(list)
for x in range(0,3):
    numerator1=0
    numerator2 = 0
    denom=0
    for y in membership.keys():
        numerator1+=pow(membership[y][x],2)*int(y.split(", ")[0])
        numerator2+= pow(membership[y][x],2) * int(y.split(", ")[1])
        denom+=pow(membership[y][x],2)
    newcentroids[x].append(numerator1/denom)
    newcentroids[x].append(numerator2/denom)

print(newcentroids)
