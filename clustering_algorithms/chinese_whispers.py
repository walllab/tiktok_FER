import numpy as np
import random
import torch
import time
from tqdm.notebook import tqdm
from itertools import permutations 

class Node():
    def __init__(self, similarity, nodeId, threshold, hard_weights):
        self.id = nodeId
        self.cluster = nodeId
        neighbors = np.where(similarity>threshold)[0]
        if len(neighbors) == 0:
            neighbors = np.argmax(similarity)
        self.neighbors = np.array(neighbors)
        self.similarities = np.array(similarity[self.neighbors])
        self.hard_weights = hard_weights

        
    def update(self, clusters):
        neighborTypes = clusters[self.neighbors]
        uniqueClusters, clusterList = np.unique(neighborTypes, return_inverse=True)

        oneHot = np.zeros((clusterList.size, clusterList.max()+1))
        oneHot[np.arange(clusterList.size),clusterList] = 1
        if self.hard_weights:
            totalSimilarity = oneHot.sum(axis=0)
        else:
            totalSimilarity = oneHot.T.dot(self.similarities)
        self.cluster = uniqueClusters[np.argmax(totalSimilarity)]
        return self.cluster

def cosin_metric_batch(x1, x2):
    x1 = torch.Tensor(x1).to('cuda:0')
    x2 = torch.Tensor(x2).to('cuda:0')  
    return torch.matmul(x1, x2).cpu().numpy()

def embedding2cosineSimilarityMatrix(encodings, tracks=None, frameNums=None):

    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []

    allDistances = cosin_metric_batch(np.array(encodings), np.array(encodings).T)

    # Maximize the similarity between the faces from same tracks
    if tracks is not None:
        uniques = np.unique(tracks)[1:]
        for unique in uniques:
            idx = np.where(tracks==unique)[0]
            if len(idx)>1:
                perm = np.array(list(permutations(idx,2)))
                allDistances[perm[:,0],perm[:,1]] = 1



    # # https://stackoverflow.com/questions/30003068/get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
    if frameNums is not None:
        idx_sort = np.argsort(frameNums)
        sorted_records_array = frameNums[idx_sort]
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,return_index=True)
        res = np.split(idx_sort, idx_start[1:])
        vals = vals[count > 1]
        duplicateIndices = filter(lambda x: x.size > 1, res)

        mutuallyExclusiveTracks = set()
        for duplist in duplicateIndices:
            # print(duplist,tracks[duplist])
            if np.all(tracks[duplist]):
                mutuallyExclusiveTracks.add(tuple(tracks[duplist]))

        for trackTuple in mutuallyExclusiveTracks:
            idx1 = np.where(tracks==trackTuple[0])[0]
            idx2 = np.where(tracks==trackTuple[1])[0]
            perm = np.array(list(permutations(idx,2)))
            allDistances[perm[:,0],perm[:,1]] = 0

    
   


def chinese_whispers(similarityMatrix, threshold, iterations=10, tracks=None, hard_weights=False, verbose=True):
    """
    similarityMatrix : nxn similarity matrix. All elements between -1 and 1. 1 for most similar
    threshold : similarity threshold
    iterations : number of iterations
    tracks : face tracks to initialize known clusters
    hard_weights : use binary weights between nodes. Converts the cluster assignment decision to majority vote

    return clustersList
    """

    random.seed(123)

    nodeList = []
    for nodeId, similarity in enumerate(similarityMatrix):
        nodeList.append(Node(similarity, nodeId, threshold, hard_weights))

    n = len(nodeList)
    clusters = np.arange(n)

    if tracks is not None:
        uniques = np.unique(tracks)[1:]
        for unique in uniques:
            idx = np.where(tracks==unique)[0]
            start, end = idx.min(), idx.max()
            clusters[start:end] = clusters[start]

    for iter in range(iterations):
        idx = random.sample(range(n), n)
        prevClusters = clusters.copy()
        
        for i in idx:
            node = nodeList[i]
            cluster = node.update(clusters)
            clusters[node.id] = cluster

        if verbose:
            print(sorted(np.unique(clusters, return_counts=True)[1], reverse=True))
        if np.all(clusters == prevClusters):
            break

    clustersList = []
    labels = np.unique(clusters)
    for label in labels: 
        clustersList.append(list(*np.where(clusters==label)))   
    # return sorted(clustersList, key=lambda x:len(x), reverse=True), clusters
    return clustersList, clusters

if __name__ == "__main__":

    similarityMatrix = np.array([[0.0, 1.0, 0.5, 0.3]
                                ,[0.0, 0.0, 0.0, 0.0]
                                ,[0.0, 0.0, 0.0, 0.6]
                                ,[0.0, 0.0, 0.0, 0.0]])
    threshold = 0.5
    print(chinese_whispers(similarityMatrix, threshold))