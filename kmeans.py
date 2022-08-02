# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import Counter

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
iris = pd.read_csv(url, encoding='utf-8')
data = iris.drop('variety', axis = 1)

#初始群中心(隨機選取k個樣本點)
k = 3
numData, dimData = data.shape   #(150,4)
def inintCentroids():
    centroid = np.zeros((k,dimData))
    for i in range(k):
        index = np.random.randint(numData, size=k)
        centroid[i,:] = data.loc[index[i],:]
    return(centroid)
print(inintCentroids())
print(type(inintCentroids()),inintCentroids().shape)

#距離計算
def eucliDist(point1,point2):
    return(np.sqrt(sum((point1-point2)**2)))

#指派群、更新群中心
def kmeans(data,k):
  global centroid
  global clusterData
  clusterData = np.array(np.zeros((numData,2))) #表每筆資[所屬群集,與該群中心之距離]
  clusterChange = True
  centroid = inintCentroids()
  while clusterChange:
    clusterChange = False
    for i in range(numData):
      minDist = float('inf') #樣本與最近群中心之距離
      clusterAssigned = 0
      for j in range(k):
        dist = eucliDist(centroid[j,:], data.loc[i,:])
        #判斷最小距離是否改變
        if dist < minDist:
          minDist = dist
          clusterData[i,1] = minDist  #更新最近群中心之距離
          clusterAssigned = j     #更新所屬群集
                    
      if clusterData[i,0] != clusterAssigned:     #迴圈重複，直到群中心不再改變
        clusterChange = True
        clusterData[i,0] = clusterAssigned
        # 更新群中心
    for j in range(k):
      clusterIndex = np.nonzero(clusterData[:,0] == j) #找出屬於每個群集的資料索引，例:群1→ data[0,4,17,....]；用nonzero判斷布林值
      pointsCluster = data.loc[clusterIndex] #每個群集所含的所有資料點
      centroid[j,:] = np.mean(pointsCluster, axis=0)
      #print(pointsCluster)
  return centroid, clusterData
print(kmeans(data,k))

#計算準確率
def accurcy():
  global accCluster
  accCluster = np.empty([3,1], dtype = int)
  compare = pd.DataFrame([clusterData[:,0], iris.iloc[:,-1]])   #col0分群後所屬群集, col1原資料之標籤
  compare = compare.T;compare.columns = ['cluster','label']
  compare = compare.sort_values('cluster')
  for j in range(k):
    clusterIndex = np.nonzero(clusterData[:,0] == j)
    pointsCluster = iris.loc[clusterIndex]
    pointsCluster = pointsCluster.iloc[:,-1]
    mostOccur = np.empty([3,1], dtype = "S12") 
    mostOccur[j,:] = pd.DataFrame([Counter(pointsCluster).most_common(1)[0][0]])
    mostOccur = mostOccur.astype(str)   #b'Virginica'為byte string
  
    accCluster[j] = len(compare[(compare['cluster'] == j) & (compare['label'] == mostOccur[j,:][0])])
    acc = (np.sum([accCluster]))/ len(iris)
  return acc
print(accurcy())