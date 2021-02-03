# How to use

## choose the dataset
there are 2 datasets included in the git repo.

 - tokenized with args
 - tokenized as instructions
 
### tokenized with args
this dataset has the largest sequence size of the 2 per sequence and each instruction and argument is represented by a token.

### tokenized as instructions
this dataset has the smallest seqence size of the 2 per seqence and each unique instruction, argument combination is represented by a token


## How to use
unzip the required dataset.

for verrifying that the dataset can be read properly 
`python3 model_trainer.py -v -c [correct config file correlating to dataset] -p path_to_dataset_folders`
It will return once slice of the dataset so the dimentions can be verrified.

for training the model:
`python3 model_trainer.py -b 64 -e 20 -c [correct config file correlating to dataset] -p path_to_dataset_folders`
the config files are included in the github and are pretty straight forward
the `path_to_dataset_folders` is the path to the folder containing the `train` and `test` folder of the required dataset.

it currently is not set to run on GPU if you want to change that. you can incapsulate the model.fit with:
```
with tf.device('/GPU:0'): # or any other gpu
	model.fit(...)
```

