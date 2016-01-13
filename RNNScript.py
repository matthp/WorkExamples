import pandas as pd
from pybrain.structure import RecurrentNetwork, FullConnection, LinearLayer, TanhLayer, LSTMLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.datasets.sequential import SequentialDataSet

# Import dataset
df_path = '/Users/admin/Desktop/Allen AI Challenge Data/training_df'
df = pd.read_pickle(df_path)

#testdf_path = '/Users/admin/Desktop/Allen AI Challenge Data/validation_df'
#testdf = pd.read_pickle(testdf_path)

# Construct one hot character dictionary
conglomerateString = ''
for index, row in df.iterrows():
    conglomerateString += row.values

conglomerateSet = list(set(list(conglomerateString[0])))
codeTable = pd.Series(data=conglomerateSet, index=conglomerateSet)
codeTable = pd.get_dummies(codeTable)

conglomerateSet = []
conglomerateString = []

# Construct LSTM network
rnn = RecurrentNetwork()

inputSize = len(codeTable['a'].values)
outputSize = 4
hiddenSize = 10

rnn.addInputModule(LinearLayer(dim=inputSize, name='in'))
rnn.addModule(TanhLayer(dim=hiddenSize, name = 'in_proc'))
rnn.addModule(LSTMLayer(dim=hiddenSize, peepholes=True, name='hidden'))
rnn.addModule(TanhLayer(dim=hiddenSize, name = 'out_proc'))
rnn.addOutputModule(SoftmaxLayer(dim=outputSize, name='out'))

rnn.addConnection(FullConnection(rnn['in'], rnn['in_proc'], name='c1'))
rnn.addConnection(FullConnection(rnn['in_proc'], rnn['hidden'], name='c2'))
rnn.addRecurrentConnection(FullConnection(rnn['hidden'], rnn['hidden'], name='c3'))
rnn.addConnection(FullConnection(rnn['hidden'], rnn['out_proc'], name='c4'))
rnn.addConnection(FullConnection(rnn['out_proc'], rnn['out'], name='c5'))

rnn.sortModules()

# Construct dataset
trainingData = SequentialDataSet(inputSize, outputSize)

for index, row in df.iterrows():
    trainingData.newSequence()
    inputSequence = list((row.values)[0])

    outputVector = [0, 0, 0, 0]
    if index == 'A':
        outputVector[0] = 1

    if index == 'B':
        outputVector[1] = 1

    if index == 'C':
        outputVector[2] = 1

    if index == 'D':
        outputVector[3] = 1

    for item in inputSequence:
        trainingData.appendLinked(codeTable[item].values, outputVector)


# Construct trainer and train network
#trainer = RPropMinusTrainer(rnn, verbose=True)
trainer = BackpropTrainer(rnn, learningrate=0.01, lrdecay=0.99, momentum=0, verbose=True, batchlearning=False, weightdecay=0)
trainer.trainUntilConvergence(trainingData, validationData=None, validationProportion=0.25, maxEpochs=10)

# Clean up memory
trainer = []
trainingData = []
df = []

# Compute predictions on test data



