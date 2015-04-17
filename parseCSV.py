import numpy as np
import math
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import csv

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


def main():
	#csv_data = np.genfromtxt ('diabetic_data.csv', delimiter=",")
	N_SAMPLES = 101766
	N_DRUGS = 23
	N_NUMERICAL = 8 # 1 column for age, the others for the numerical values

	# the three lable vectors
	# Readmission label: 3 classes:
	# 1 for No readimission
	# 2 for <30 days
	# 3 for >30 days
	label_readmission = []

	# HBA1C label: 
	# 4 classes:
	# 1 for >8
	# 2 for >7
	# 3 for Norm (<7)
	# 4 for None
	label_HBA1C = []
	
	# Primary Diagonosis Label
	# 9 classes:
	#A disease of the circulatory system
	#Diabetes
	#A disease of the respiratory system
	#Diseases of the digestive system
	#Injury and poisoning
	#Diseases of the musculoskeletal system and connective tissue
	#Diseases of the genitourinary system
	#Neoplasms
	#Other
	label_diag1 = []

	# Medication change label: binary, 0 for NO, 1 for Yes
	label_medication_change = []


	# Numerical Data Array
	data_numerical = np.zeros(shape = (N_SAMPLES, N_NUMERICAL), dtype= np.float64)
	# Bag of Drugs
	data_bagOfDrugs = np.zeros(shape = (N_SAMPLES, N_DRUGS), dtype = int)
	# Bag of Drug Dict
	bagOfDrugs_Dict ={
	'Down':1,
	'Up':1,
	'Steady':1,
	'No':0
	}

	with open('diabetic_data.csv') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		i = 0
		# skip the first iteration
		iterrows = iter(reader)
		next(iterrows)
		for row in iterrows:
			# parse Age	
			age = int(row[4][1:row[4].find('-')])
			data_numerical[i][0] = age/10 # age class is just age/10, TODO: try with just 0,10,20,30...etc year old categories
			# parse Time in the hospital
			data_numerical[i][1] = int (row[9])
			# parse the rest of the numerical data
			data_numerical[i][2:] = [int(num) for num in row [12:18]]

			#parse bag of drugs
			drug_dosage = row[24:47]
			for j in range (0, len(drug_dosage)):
				data_bagOfDrugs[i][j] = bagOfDrugs_Dict[drug_dosage[j]]
			i+= 1

	print data_numerical[-1:,] # see the last patient for verification
	print data_bagOfDrugs

	#print np.asarray(label_readmission)
	return
if __name__ == "__main__":
	main()
