import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import grid_search
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

def readAndPreProcess():
	print("\n\n********** CS-412 HW5 Mini Project **********")
	print("************ Submitted by Sankul ************\n\n")
	print("Reading data, please ensure that the dataset is in same folder.")
	resp = pd.read_csv('responses.csv')
	print("Data reading complete!")
	print("Some stats reagarding data:")
	resp.describe()
	
	print("\nStarting pre-processing.....")
	
	print("\nFinding missing values:")
	print("Missing values found, removing them")
	emptyVals = resp.isnull().sum().sort_values(ascending=False)
	emptyPlot = emptyVals.plot(kind='barh', figsize = (20,35))
	plt.show()
	print("Empty values removed")
	
	print("\nChecking for NaN and infinite values in target column (Empathy):")
	if len(resp['Empathy']) - len(resp[np.isfinite(resp['Empathy'])]):
		print("Number of infinite or NaN values in Empathy column: ", len(resp['Empathy']) - len(resp[np.isfinite(resp['Empathy'])]))
		print("Removing them")
		resp = resp[np.isfinite(resp['Empathy'])]
		print("Infinite and NaN values removed")
		
	print("\nChecking for categorical features:")
	if pd.Categorical(resp).dtype.name == 'category':
		print("Categorical features found. Removing them...")
		resp = resp.select_dtypes(exclude=[object])	
		print("Categorical features removed")
		
	print("\nReplacing NaN values with the mean value:")
	resp=resp.fillna(resp.mean()) 
	resp.isnull().sum()
	print("Values replaced")
	
	print("\nSeperating labels from data:")
	Y = resp['Empathy'].values
	X = resp.drop('Empathy',axis=1)
	print("Labels seperated")
	
	print("\nScaling, standardizing and normalizing the data:")
	scaler = MinMaxScaler(feature_range=(0, 1))
	rescaledX = scaler.fit_transform(X)
	
	scaler = StandardScaler().fit(rescaledX)
	standardizedX = scaler.transform(rescaledX)
	
	normalizer = Normalizer().fit(standardizedX)
	normalizedX = normalizer.transform(standardizedX)
	print("Scaling, standardizing and normalizing completed")
	
	print("\nFinal data looks like:")
	print(normalizedX.shape)
	print("Values inside look like:")
	print(normalizedX[0])
	
	return normalizedX,Y

def train(normalizedX, Y):
	print("\nSplitting data in Train(90%), Development(10%) and Test(10%):")
	X_t, X_te, y_t, y_te = train_test_split(normalizedX, Y, test_size=0.1, random_state=1)
	X_t, X_v, y_t, y_v = train_test_split(X_t, y_t, test_size=0.1, random_state=1)
	print("Splitting complete")
	print("Data shape before: ",normalizedX.shape)
	print("Label shape before: ",normalizedX.shape)
	print("Train data shape: ",X_t.shape)
	print("Train label shape: ",y_t.shape)
	print("Development data shape: ",X_v.shape)
	print("Development label shape: ",y_v.shape)
	print("Test data shape: ",X_te.shape)
	print("Test label shape: ",y_te.shape)
	
	print("\nTraining base classifiers, with cross validation")
	
	accList = []
	clfList = []
	
	print("\nGaussian Naive Bayes:")
	predictedNB = cross_val_predict(GaussianNB(), X_t, y_t, cv=3)
	print(classification_report(y_t, predictedNB))
	print("The accuracy score is {:.2%}".format(metrics.accuracy_score(y_t, predictedNB)))
	accList.append(metrics.accuracy_score(y_t, predictedNB))
	clfList.append("Gaussian Naive Bayes")
	
	print("\nDecision Tree:")
	predictedDT = cross_val_predict(DecisionTreeClassifier(), X_t, y_t, cv=3)
	print(classification_report(y_t, predictedDT))
	print("The accuracy score is {:.2%}".format(metrics.accuracy_score(y_t, predictedDT)))
	accList.append(metrics.accuracy_score(y_t, predictedDT))
	clfList.append("Decision Tree")
	
	print("\nKNN:")
	predictedKNN = cross_val_predict(KNeighborsClassifier(), X_t, y_t, cv=3)
	print(classification_report(y_t, predictedKNN))
	print("The accuracy score is {:.2%}".format(metrics.accuracy_score(y_t, predictedKNN)))	
	accList.append(metrics.accuracy_score(y_t, predictedKNN))
	clfList.append("KNN")
	
	print("\nSVM:")
	predictedSVC = cross_val_predict(SVC(), X_t, y_t, cv=3)
	print(classification_report(y_t, predictedSVC))
	print("The accuracy score is {:.2%}".format(metrics.accuracy_score(y_t, predictedSVC)))
	accList.append(metrics.accuracy_score(y_t, predictedSVC))
	clfList.append("SVM")
	
	print("\nRandom Forest:")
	predictedRF = cross_val_predict(RandomForestClassifier(), X_t, y_t, cv=3)
	print(classification_report(y_t, predictedRF))
	print("The accuracy score is {:.2%}".format(metrics.accuracy_score(y_t, predictedRF)))
	accList.append(metrics.accuracy_score(y_t, predictedRF))
	clfList.append("Random Forest")
	
	print("Multi-Layer Perceptron:")
	predictedMLP = cross_val_predict(MLPClassifier(), X_t, y_t, cv=3)
	print(classification_report(y_t, predictedMLP))
	print("The accuracy score is {:.2%}".format(metrics.accuracy_score(y_t, predictedMLP)))
	accList.append(metrics.accuracy_score(y_t, predictedMLP))
	clfList.append("Multi-Layer Perceptron")
	
	print("\nPicking the best classifier according to accuracy:")
	plt.plot(clfList,accList,'ro')
	plt.ylabel('Accuracy')
	plt.xlabel('Classifier')
	plt.show()	
	print("Best classifier is : ", clfList[accList.index(max(accList))])
	print("It's accuracy : ", max(accList))
	
	print("\n\nTuning hyperparameter of best classifier:")
		
	print("\nRunning grid_search...")
	parameters = {'max_iter':np.arange(200,500,50)}
	clfMLPGS = grid_search.GridSearchCV(MLPClassifier(), parameters, cv=10)
	clfMLPGS.fit(X_t, y_t)
	predictedMLPGS = clfMLPGS.predict(X_v)

	print("We get the classifier with best parameters")
	clfMLP = MLPClassifier(max_iter = 300, activation='logistic', learning_rate='invscaling', random_state=1).fit(X_t,y_t) 
	predictedMLP = clfMLP.predict(X_v)  
	print("Accuracy after tuning : ", np.mean(predictedMLP == y_v))
	
	print("\nFinding the K best features:")
	maxK = 0
	maxAcc = 0
	acc = []

	for ft in range(1,139,5):
		feat_selector = SelectKBest(f_classif, k=ft)
		X_new = feat_selector.fit_transform(X_t, y_t)
		clfMLP = MLPClassifier(max_iter = 300, activation='logistic', learning_rate='invscaling', random_state=1).fit(X_new,y_t) 
		predictedMLP = clfMLP.predict(feat_selector.transform(X_v))
		acc.append(np.mean(predictedMLP == y_v))
		
		if maxAcc < np.mean(predictedMLP == y_v):
			maxK = ft
			maxAcc = np.mean(predictedMLP == y_v)
			
	plt.plot(acc)
	plt.ylabel('Accuracy')
	plt.xlabel('K/5  (5 is the iterator)')
	plt.show()

	print("Best values: ")
	print("K = ", maxK)
	print("Accuracy = ", maxAcc)
	
	
	print("\n\nTraining final classifier with tuned hyperparameters and K best features.....")
	feat_selector = SelectKBest(f_classif, k=maxK)
	X_newFinal = feat_selector.fit_transform(X_t, y_t)
	clfMLP = MLPClassifier(max_iter = 300, activation='logistic', learning_rate='invscaling', alpha=0.1, learning_rate_init=0.01, shuffle=False, random_state=1).fit(X_newFinal,y_t) 
	print("Classifier trained")
	print("Running on test data.......")
	predictedMLP = clfMLP.predict(feat_selector.transform(X_te))
	print("Test accuracy = ", np.mean(predictedMLP == y_te))
	
	
	
	
preProcessedData, labels = readAndPreProcess()
train(preProcessedData, labels)