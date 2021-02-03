import tensorflow as tf 
from tensorflow import keras
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-b','--batchSize', type=int, help="Batch size")
parser.add_argument('-e','--epochs', type=int, help="training_epochs")
parser.add_argument('-p','--path', type=str, help="basepath to dataset")
parser.add_argument('-c','--config', type=str, help="path to config file")
parser.add_argument('-v', '--verrify', action='store_true' ,help="verrify data path")

def train_generator(batch_size,basePath, sample_number):
	path = basePath + "/train/"
	batchcounter =0
	data = []
	labels = []
	file_template = path+"{}.json"
	for i in range(sample_number):
		with open(file_template.format(i), 'r') as f:
			jdata = json.load(f)
		data.append(jdata['data'])
		labels.append(jdata['key'])
		batchcounter+= 1
		if batchcounter >= batch_size or i == sample_number-1:
			x = np.array(data, dtype='float32')
			y = np.array(labels, dtype='float32')
			yield (x,y)
			data = []
			labels = []
			batchcounter = 0

def test_generator(batch_size,basePath, sample_number):
	path = basePath + "/test/"
	batchcounter =0
	data = []
	labels = []
	file_template = path+"{}.json"
	for i in range(sample_number):
		with open(file_template.format(i), 'r') as f:
			jdata = json.load(f)
		data.append(jdata['data'])
		labels.append(jdata['key'])
		batchcounter+= 1
		if batchcounter >= batch_size or i == sample_number-1:
			x = np.array(data, dtype='float32')
			y = np.array(labels, dtype='float32')
			yield (x,y)
			data = []
			labels = []
			batchcounter = 0



def MalMem(vocab_size, emb_size,net_size, output_pars, max_len):
	model = keras.Sequential([
		keras.layers.Embedding(vocab_size, emb_size, input_length=max_len),
		keras.layers.Bidirectional(keras.layers.GRU(net_size)),
		keras.layers.Dense(emb_size/2, activation='relu'),
		keras.layers.Dense(output_pars, activation='sigmoid')
		])
	return model


def import_config(filename):
	with open(filename, 'r') as f:
		jdata = json.load(f)

	max_len = jdata["max_len"]
	train_len = jdata["train_len"]
	test_len = jdata["test_len"]

	return train_len, test_len, max_len


if __name__ == "__main__":
	args = parser.parse_args()
	train_len, test_len, max_len = import_config(args.config)
	if args.verrify:
			a = train_generator(5, args.path ,train_len)
			print(next(a)[0].shape)
			exit()

	epochs = args.epochs
	batches = args.batchSize
	out= """
	epochs: {}
	baches: {}
	train lenght: {}
	test length: {}
	max size: {}
	"""
	print(out.format(epochs,batches,train_len,test_len, max_len))

	####################ACTUAL MODEL FITTING##################
	model = MalMem(100, 150, 128, 12, 292330)
	model.compile(optimizer='Adam', loss='mse')
	print(model.summary())
	model.fit(train_generator(batches, args.path ,train_len),verbose=1, validation_data=test_generator(batches, args.path, test_len), batch_size=batches, epochs=epochs)

