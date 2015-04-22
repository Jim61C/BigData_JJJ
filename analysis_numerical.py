import numpy as np
import math
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from sklearn import preprocessing

from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import feature_selection as fs

sigma = None

def saveSparse (array, filename):
	np.savez(filename,data = array.data ,indices=array.indices,indptr =array.indptr, shape=array.shape )

def loadSparse(filename):
	loader = np.load(filename)
	return csc_matrix((  loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def saveArray (array, filename):
	np.savez(filename,data = array)

def loadArray(filename):
	loader = np.load(filename)
	return np.array(loader['data'])

def KNearest_Neighbour(X_train, y_train, X_test, y_test, num_neighbours):
	neigh = KNeighborsClassifier(n_neighbors=num_neighbours) # n neighbours = 1,3,5, default distance is standard Euclidean distance, weights in prediction is uniform
	neigh.fit(X=X_train, y=y_train)
	#print "K nearest neighbour prediction result:", neigh.predict(X_test)
	#print "True test label:", y_test
	return neigh.score(X= X_test, y=y_test)

def KNearest_Neighbour_Weighted_Similarity(X_train, y_train, X_test, y_test, num_neighbours):
	neigh = KNeighborsClassifier(n_neighbors=num_neighbours, weights = DistanceArrayToSimilarityArray) # n neighbours = 1,3,5, default distance is standard Euclidean distance, weights is the similarity function
	neigh.fit(X=X_train, y=y_train)
	#print "K nearest neighbour prediction result:", neigh.predict(X_test)
	#print "True test label:", y_test
	return neigh.score(X= X_test, y=y_test)

def DistanceArrayToSimilarityArray(arr):
	global sigma
	print "sigma in DistanceArrayToSimilarityArray function is:", sigma
	if(sigma == None):
		raise ValueError("Global sigma is not set!")
	K = np.exp(-1 * arr ** 2 / (2* sigma ** 2))
	#print K
	return K # k is of shape (N_sampels, N_samples)

#def DistanceValueToSimilarityValue(distance):
#	global sigma
#	if (sigma == None):
#		sigma = 500 # the best value for Raw data
#	return math.exp(-1*math.pow(distance,2)/(math.pow(sigma,2)*2))

def KNearestNeighbour_Analysis(data, label_column, num_folds, num_neighbours): # data is of shape ( N_samples, N_features), label_column is(N_samples, )
	# 5 fold or 10 fold
	accuracies = []
	kf = StratifiedKFold(label_column, n_folds=num_folds) # need stratified KF, otherwise, the normal KF just picks the first 80 as a subsample, 80-160 as another subsample -> in this case, the test data's label info won't be even in the training data set
	for train, test in kf:
		#print "train index:" , train
		#print "test index:", test
		X_train, X_test, y_train, y_test = data[train], data[test], label_column[train], label_column[test]
		#print "X_train:", X_train.shape
		#print "y_train:", y_train.shape
		#print "X_test:",  X_test.shape
		#print "y_test:", y_test.shape,"\n"
		accuracy  = KNearest_Neighbour(X_train, y_train, X_test, y_test, num_neighbours = num_neighbours)
		#accuracy  = KNearest_Neighbour_Weighted_Similarity(X_train, y_train, X_test, y_test, num_neighbours = num_neighbours)
		#print "The accuracy:", accuracy
		accuracies.append(accuracy)
	global sigma
	print "The mean of the accuracies:", np.mean(accuracies), "; The standard Deviation of the accuracies:", np.std(accuracies, dtype=np.float64), "" if (sigma == None) else "with sigma tuned as {sigma}".format(sigma= sigma)
	return np.mean(accuracies)


def SVM_Linear (data_train, label_train, data_test, label_test):
	lin_svm = svm.LinearSVC()
	lin_svm.fit(data_train, label_train)
	return lin_svm.score(data_test, label_test)

def SVM_SVC(data_train, label_train, data_test, label_test):
	#svc_svm = svm.SVC(kernel='linear')
	svc_svm = svm.SVC(dual = False)
	svc_svm.fit(data_train, label_train)
	return svc_svm.score(data_test, label_test)

def SVM_Analysis (data, label_column, num_folds, SVMToUse):
	accuracies = []
	kf = StratifiedKFold(label_column, n_folds=num_folds) # need stratified KF, otherwise, the normal KF just picks the first 80 as a subsample, 80-160 as another subsample -> in this case, the test data's label info won't be even in the training data set
	for train, test in kf:
		#print "train index:" , train
		#print "test index:", test
		X_train, X_test, y_train, y_test = data[train], data[test], label_column[train], label_column[test]
		#print "X_train:", X_train.shape
		#print "y_train:", y_train.shape
		#print "X_test:",  X_test.shape
		#print "y_test:", y_test.shape,"\n"
		accuracy  = SVMToUse(X_train, y_train, X_test, y_test)
		print "The accuracy for one fold run:", accuracy
		accuracies.append(accuracy)
	global sigma
	print "The mean of the accuracies:", np.mean(accuracies), "; The standard Deviation of the accuracies:", np.std(accuracies, dtype=np.float64), "" if (sigma == None) else "with sigma tuned as {sigma}".format(sigma= sigma)
	return np.mean(accuracies)

def featureSelection_Linear_SVM (data, label, penalty): # data in shape(N_samples, N_feature), label is a 1D vector (N_samples,)
	svc_svm = svm.LinearSVC(penalty = 'l1', dual= False, C = penalty) # Tune the C for sparsity 
	svc_svm.fit(data, label)
	selected_data = svc_svm.transform(data)
	print "The selected shape is:", selected_data.shape
	return svc_svm.coef_


def main():
	data_numerical = loadArray("data_numerical.npz")
	#label_readmission = loadArray("label_readmission.npz")
	label_readmission = loadArray("label_readmission_two_class.npz")

	#Normalize
	normalized_data_numerical = preprocessing.normalize(X = data_numerical, axis = 0, norm ='l2')
	#Standardize
	standardizer = preprocessing.StandardScaler().fit(data_numerical)
	standardized_data_numerical = standardizer.transform(data_numerical)

	for i in range (0, standardized_data_numerical.shape[1]):
		print np.mean(standardized_data_numerical[:,i])
		print np.std(standardized_data_numerical[:,i])

	#for i in range (0, normalized_data_numerical.shape[1]):
	#	print np.linalg.norm(normalized_data_numerical[:,i],2)

	#------------------K nearest Neighbour------------------
	#print "--------------K nearest neighbours analysis on Numerical Data----------------"
	#folds = [5]
	#neighbours = [7,15,30]
	#for i in range (0, len(folds)):
	#	for j in range(0, len(neighbours)):
	#		print "Normalized, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
	#		KNearestNeighbour_Analysis(data = normalized_data_numerical, label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j])
	#		print "Standardized, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
	#		KNearestNeighbour_Analysis(data = standardized_data_numerical, label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j])

	#------------------- SVM ------------------------------
	#print "------------------- SVM on Numerical Data normalized------------------------------"
	#SVM_Analysis(data = normalized_data_numerical, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear)
	#SVM_Analysis(data = normalized_data_numerical, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_SVC)


	data_bag_drugs = loadSparse("data_bagOfDrugs_sparse.npz")
	selected_column = []
	for i in range(0, data_bag_drugs.toarray().shape[1]):
		#print "if", i, "th column is all zeros:", (data_bag_drugs.toarray()[:,i] == 0).all()
		if(not (data_bag_drugs.toarray()[:,i] == 0).all()):
			selected_column.append(True)
		else:
			selected_column.append(False)
		#print "column sum is:", np.sum(data_bag_drugs.toarray()[:,i])

	print "after truncate the zero column features for bag of drugs:", data_bag_drugs[:,np.array(selected_column) == True].shape
	data_bag_drugs = data_bag_drugs[:,np.array(selected_column) == True]
	#Normalize
	normalized_bag_drugs = preprocessing.normalize(X = data_bag_drugs.toarray().astype(float), axis = 0, norm = 'l2')
	#standardize
	standardized_bag_drugs = preprocessing.StandardScaler().fit(data_bag_drugs.toarray().astype(float)).transform(data_bag_drugs.toarray().astype(float))


	#for i in range(0, normalized_bag_drugs.shape[1]):
	#	print i,"th column mean of standardized:", np.mean(standardized_bag_drugs[:,i])
	#	print i, "th column sd of standardized:", np.std(standardized_bag_drugs[:,i])
	#	print i, "th column norm:", np.linalg.norm(normalized_bag_drugs[:,i], 2)


	#print "--------------K nearest neighbours analysis on Bag of Drugs----------------"
	#folds = [5]
	#neighbours = [7,15,30]
	#for i in range (0, len(folds)):
	#	for j in range(0, len(neighbours)):
	#		print "Normalized, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
	#		KNearestNeighbour_Analysis(data = normalized_bag_drugs, label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j])
	#		print "Standardized, num_folds = ", folds[i], ", num_neighbours = ", neighbours[j]
	#		KNearestNeighbour_Analysis(data = standardized_bag_drugs, label_column = label_readmission, num_folds = folds[i], num_neighbours = neighbours[j])

	#print "------------------- SVM on Bag of Drugs Normalized------------------------------"
	#SVM_Analysis(data = normalized_bag_drugs, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_Linear)
	#SVM_Analysis(data = normalized_bag_drugs, label_column = label_readmission,num_folds = 5, SVMToUse = SVM_SVC)



	#---------------Feature selection-----------------
	print "rank of numerical features:", featureSelection_Linear_SVM(data = normalized_data_numerical, label = label_readmission, penalty = 0.08)
	print "rank of drugs:", featureSelection_Linear_SVM(data = data_bag_drugs.toarray().astype(float), label = label_readmission, penalty = 0.01)
	
	estimator = svm.LinearSVC(penalty = 'l1', dual = False)
	rfe = fs.RFE(estimator = estimator, n_features_to_select = 3)
	rfe.fit(normalized_data_numerical, label_readmission)
	print "rfe on numerical support_:",rfe.support_
	print "rfe on numerical ranking_:", rfe.ranking_







	return


if __name__ == "__main__":
	main()
