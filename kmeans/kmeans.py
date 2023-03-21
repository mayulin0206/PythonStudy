'''
k-means工作流程
1.首先确定k个初始点作为质心（一般不是数据中的点）
2.随机将数据中的每个点分配到一个簇中，具体的讲，就是为每个点找到距离其最近的质心，并分配到簇中
这一步完成后，每个质心更新为该簇所有点平均值

创建k个点作为初始质心（通常随机选择）
当任意一个点簇的分配结果发生变化时
	对数据中的每个数据点
		对每个质心
			计算质心到数据点之间的距离
		将数据点分配到距离其最近的核
	对每个簇，计算簇中所有点的均值作为其簇核心

'''
import os
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
def loadDataSet(filename):
	data = np.loadtxt(filename,delimiter = '\t')
	return data;

# 欧氏距离计算
def distEclud(x,y):
	return np.sqrt(np.sum((x - y)**2))

# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,k):
	m,n = dataSet.shape
	cent = np.zeros((k,n))
	for i in range(k):
		index = int(np.random.uniform(0,m))
		cent[i,:] = dataSet[index,:]
	return cent;

# k均值聚类
def kMeans(dataSet,k):
	#行的数目
	m = np.shape(dataSet)[0]
	# 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
	clusterAssment = np.mat(np.zeros((m,2)))
	clusterChange = True
	
	# 第1步 初始化centroids
	cent = randCent(dataSet,k)
	while clusterChange:
		clusterChange = False
		# 遍历所有的样本（行数）
		for i in range(m):
			minDist = 1000000.0
			minIndex = -1

			# 遍历所有的质心
			#第2步 找出最近的质心
			for j in range(k):
				# 计算该样本到质心的欧式距离
				distance = distEclud(cent[j,:],dataSet[i,:])
				if distance < minDist:
					minDist = distance
					minIndex = j

			# 第 3 步：更新每一行样本所属的簇
			if clusterAssment[i,0] != minIndex:
				clusterChange = True
				clusterAssment[i,:] = minIndex,minDist**2

		#第 4 步：更新质心
		for j in range(k):
			# 获取簇类所有的点
			pointInCluster = dataSet[np.nonzero(clusterAssment[:,0].A ==j)[0]]
			# 对矩阵的行求均值
			cent[j,:] = np.mean(pointInCluster,axis = 0)

	print("Congratulation")
	return cent,clusterAssment

def showCluster(dataSet,k,cent,clusterAssment):
	m,n = dataSet.shape
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	# 绘制所有的样本
	for i in range(m):
		plt.plot(dataSet[i,0],dataSet[i,1],mark[int(clusterAssment[i,0])])
	# 绘制质心
	for i in range(k):
		plt.plot(cent[i,0],cent[i,1],mark[i + 4])
	plt.show()	

if __name__ == '__main__':
	dataSet = loadDataSet("kmeans.txt")
	k = 4
	centroids,clusterAssment = kMeans(dataSet,k)
	showCluster(dataSet,k,centroids,clusterAssment)
	'''
	A = np.mat(np.zeros((2,2)))
	A[1,0] = 2
	print(A)
	print(np.nonzero(A[:,0].A==2)[0])
	'''