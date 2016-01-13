from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.utils import resample
from sknn.mlp import Regressor, Layer, Classifier
import pandas as pd
import numpy as np
from sklearn import metrics


def calcAUC(prediction, truth):
    fpr, tpr, thresholds = metrics.roc_curve(truth, prediction, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    return AUC

trainingDataPath = '/Users/admin/Desktop/numerai_datasets/numerai_training_data.csv'

data = pd.read_csv(trainingDataPath)

# indices of validation examples
i = data.validation == 1

val = data[i].copy()
train = data[~i].copy()

# no need for the column anymore
train.drop('validation', axis=1, inplace=True)
val.drop('validation', axis=1, inplace=True)

# create training data
train_dummies = pd.get_dummies(train.c1)
trainData = pd.concat((train.drop('c1', axis=1), train_dummies), axis=1)
trainLabels = trainData.target.as_matrix()
trainData.drop('target', axis=1, inplace=True)
trainData = trainData.as_matrix()

val_dummies = pd.get_dummies(val.c1)
valData = pd.concat((val.drop('c1', axis=1), val_dummies), axis=1)
valLabels = valData.target.as_matrix()
valData.drop('target', axis=1, inplace=True)
valData = valData.as_matrix()

# scale data
#transformer = StandardScaler()
#trainData = transformer.fit_transform(trainData)
#valData = transformer.transform(valData)

transformer = PolynomialFeatures()
trainData = transformer.fit_transform(trainData)
valData = transformer.transform(valData)

transformer = StandardScaler()
trainData = transformer.fit_transform(trainData)
valData = transformer.transform(valData)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////

# Train random forest regressor

numModels = 10
oldTrainData = trainData
oldValData = valData

#for n in range(numModels):
#    model = RandomForestRegressor(n_estimators=1, min_samples_leaf=500, n_jobs=6, random_state=n)
#    model.fit(oldTrainData, trainLabels)
#    trainData = np.column_stack((trainData, model.predict(oldTrainData)))
#    valData = np.column_stack((valData, model.predict(oldValData)))
#    print n

# Train neural network
#nn = Classifier(
#    layers=[
#        Layer("Tanh", units=200),
        #Layer("Tanh", units=100),
        #Layer("Maxout", units=150, pieces=5),
#        Layer("Softmax")],
#    learning_rate=0.005,
#    n_iter=100, verbose=True, dropout_rate=0.1, batch_size=200, random_state=5, valid_size=0.5, learning_momentum=0)
#nn.fit(trainData, trainLabels)

nn = GradientBoostingClassifier(n_estimators=100, min_samples_leaf=50, random_state=5, verbose=True, subsample=0.66)
nn.fit(trainData, trainLabels)

#nn = RandomForestClassifier(n_estimators=2000, min_samples_leaf=50, n_jobs=6, random_state=5, verbose=True)
#nn.fit(trainData, trainLabels)

# Print results

#trainPred = nn.predict(trainData)
trainPred = (nn.predict_proba(trainData))[:, 1]
trainAUC = calcAUC(trainPred, trainLabels)
print 'TrainAUC: ' + str(trainAUC)

#valPred = nn.predict(valData)
valPred = (nn.predict_proba(valData))[:, 1]
valAUC = calcAUC(valPred, valLabels)
print 'ValAUC: ' + str(valAUC)

# Predict on competition data

competitionDataPath = '/Users/admin/Desktop/numerai_datasets/numerai_tournament_data.csv'

tourn = pd.read_csv(competitionDataPath)

tourn_dummies = pd.get_dummies(tourn.c1)
tournData = pd.concat((tourn.drop('c1', axis=1), tourn_dummies), axis=1)
tournIDs = tournData.t_id.as_matrix()
tournData.drop('t_id', axis=1, inplace=True)
tournData = tournData.as_matrix()
















