#!/usr/bin/env python
# coding: utf-8


# Install dependencies

# !pip3 install -r requirements.txt -U
# !python3 -m spacy download en_core_web_md

# Suppress warnings

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# Import core lib

# In[ ]:


import sys
sys.path.append('./core')
from models.predicate_extractor import PredicateExtractor


# Import dependencies

# In[ ]:


import numpy as np
import pandas as pd
import pickle
import re
import os
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import json
#import random


# In[ ]:


import ray
import ray.tune as tune
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
import multiprocessing


# In[ ]:


import tf_metrics
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sklearn.preprocessing as preprocessing
from imblearn import over_sampling, under_sampling, combine


# Define constants

# In[ ]:


USE_SCICITE = True
USE_ACLARC = True
USE_PATTERN_EMBEDDING = True
USE_TEST_SET = False

ZERO_CLASS = 'none'
LABELS_TO_EXCLUDE = [
	#'cites',
	'cites_as_review',
	#'extends', 
	#'uses_data_from', 
	#'uses_method_in',
]

TRAIN_EPOCHS = None
MAX_STEPS = 10**4
EVALUATION_PER_TRAINING = 30
EVALUATION_STEPS = MAX_STEPS/EVALUATION_PER_TRAINING
MODEL_DIR = './model'
DATA_DIR = './dataset'
TF_MODEL = 'USE_MLQA'
MODEL_OPTIONS = {'tf_model':TF_MODEL, 'use_lemma':False}


# Define function for extracting probabilities from scicite models

# In[ ]:


def fuse_with_scicite_model(df, dataset_file, model_name):
	filename = dataset_file+'.'+model_name+'.json'
	if not os.path.isfile(filename):
		return df
	print(f'Reading {filename}..')
	extra_df = pd.read_json(filename)
	extra_df = extra_df[['probabilities','string']]
	feature_name = model_name+'_prediction'
	extra_df = extra_df.rename(columns={
		'probabilities': feature_name, 
		'string': 'anchorsent',
	})
	df = pd.merge(extra_df, df, on='anchorsent', how='inner', sort=True).drop_duplicates(subset=['anchorsent'])
	class_size = max(map(lambda x: len(x), filter(lambda x: type(x) in [list,tuple,np.array,np.ndarray], df[feature_name].to_list())))
	print(f'{model_name} has class size {class_size}')
	df[feature_name] = df[feature_name].map(lambda x: np.zeros(class_size)+1/class_size if not(type(x) in [list,tuple,np.array] and len(x)== class_size) else x)
	for e in df[feature_name].to_list():
		assert(len(e)==class_size)
	return df


# Define function for converting input datasets from csv to pandas dataframe

# In[ ]:


def get_dataframe(dataset_file):
	# Load dataset
	df = pd.read_csv(dataset_file+'.csv', sep='	')
	#print(df.dtypes)

	# Get target values list
	df['citfunc'].replace(np.NaN, 'none', inplace=True)
	df['citfunc'] = df['citfunc'].map(lambda x: x.strip())
	# Remove rows with excluded labels
	for label in LABELS_TO_EXCLUDE:
		df.loc[df.citfunc == label, 'citfunc'] = ZERO_CLASS
	# Remove bad rows
	df['citfunc'].replace('ERROR', 'none', inplace=True)
	df = df[df.citfunc != 'none']

	# Extract features from dataframe
	df = df[['anchorsent','sectype','citfunc']]
	
	# Remove null values
	df['anchorsent'].replace(np.NaN, '', inplace=True)
	df = df[df.anchorsent != '']
	df['sectype'].replace(np.NaN, 'none', inplace=True)

	df['anchorsent'] = df['anchorsent'].map(lambda x: re.sub(r'\[\[.*\]\]','',x))
	df['anchorsent'] = df['anchorsent'].map(lambda x: re.sub(r'[^\x00-\x7F]+',' ',x))
	df['anchorsent'] = df['anchorsent'].map(lambda x: re.sub(r"^'(.*)'$",r'\1',x))
    
	# Join with scicite output
	if USE_SCICITE:
		df = fuse_with_scicite_model(df, dataset_file, 'scicite')
        
	# Join with ac_larc output
	if USE_ACLARC:
		df = fuse_with_scicite_model(df, dataset_file, 'aclarc')

	# Print dataframe
	print('Dataframe')
	print(df)
	
	# Return dataset
	df.drop_duplicates(subset=['anchorsent'], inplace=True)
	y_list = df.pop('citfunc').values.tolist() # Extract target list
	feature_list = df.columns.values.tolist()
	x_dict = {feature: df[feature].to_list() for feature in feature_list}
	return {'x':x_dict, 'y':y_list}


# Define function for casting dataset to numpy arrays

# In[ ]:


def numpyfy_dataset(set):
	set['x'] = {k:np.array(v) for k,v in set['x'].items()}
	set['y'] = np.array(set['y'])


# Define function for encoding a dataset, from string to numerical representations

# In[ ]:


def encode_dataset(dataset):
	# Embed anchor sentences into vectors
	for key,value in dataset.items():
		df = value['x']
		if USE_PATTERN_EMBEDDING:
			df['main_predicate'] = df['anchorsent']
		# Embed anchor sentences
		#df['anchorsent'] = list(df['anchorsent'])
		cache_file = f'cache/{TF_MODEL}.{key}.anchorsent.embedding_cache.pkl'
		if os.path.isfile(cache_file):
			with open(cache_file, 'rb') as f:
				embedded_sentences_dict = pickle.load(f)
				embedded_sentences = [embedded_sentences_dict[s] for s in df['anchorsent']]
		else:
			MODEL_MANAGER = PredicateExtractor(MODEL_OPTIONS)
			embedded_sentences = MODEL_MANAGER.embed(df['anchorsent'])
			with open(cache_file, 'wb') as f:
				pickle.dump(dict(zip(df['anchorsent'],embedded_sentences)), f)
		df['anchorsent_embedding'] = embedded_sentences
		# Embed extra info
		if USE_PATTERN_EMBEDDING:
			cache_file = f'cache/{TF_MODEL}.{key}.extra.embedding_cache.pkl'
			if os.path.isfile(cache_file):
				with open(cache_file, 'rb') as f:
					embedded_extra_dict = pickle.load(f)
					embedded_extra = [embedded_extra_dict[s] for s in df['main_predicate']]
			else:
				MODEL_MANAGER = PredicateExtractor(MODEL_OPTIONS)
				extra_list = []
				for text in df['main_predicate']:
					extra = list(Counter(pattern['predicate'] for pattern in MODEL_MANAGER.get_pattern_list(text)).keys())
					extra_list.append(extra[0] if len(extra)>0 else '')
				embedded_extra = MODEL_MANAGER.embed(extra_list)
				with open(cache_file, 'wb') as f:
					pickle.dump(dict(zip(df['main_predicate'],embedded_extra)), f)
			df['main_predicate'] = embedded_extra

	# Encode labels
	label_encoder_target = LabelEncoder()
	label_encoder_target.fit([e for set in dataset.values() for e in set['y']])
	print('Label classes:', list(label_encoder_target.classes_))
	for set in dataset.values():
		set['y'] = label_encoder_target.transform(set['y'])

	# Encode sectypes
	all_sectypes = [e for set in dataset.values() for e in set['x']['sectype']]
	label_encoder_sectype = LabelEncoder()
	all_sectypes = label_encoder_sectype.fit_transform(all_sectypes)
	onehot_encoder_sectype = OneHotEncoder()
	onehot_encoder_sectype.fit(all_sectypes.reshape(-1, 1))
	print('SCAR classes:', list(label_encoder_sectype.classes_))
	for set in dataset.values():
		labeled_sectypes = label_encoder_sectype.transform(set['x']['sectype'])
		set['x']['sectype'] = onehot_encoder_sectype.transform(labeled_sectypes.reshape(-1, 1)).toarray()[:,1:]

	# Input features to numpy array
	for set in dataset.values():
		numpyfy_dataset(set)
	# Return number of target classes
	return len(label_encoder_target.classes_)


# Define function for resampling the dataset

# In[ ]:


def resample_dataset(set, resampling_fn=None):
	if resampling_fn is None:
		return
	#numpyfy_dataset(set)
	print('Dataset size before re-sampling:', len(set['y']))

	# Build combined features
	combined_features_sizes = {}
	combined_features_list = []
	for feature in zip(*set['x'].values()):
		combined_features = []
		for e,data in enumerate(feature):
			if type(data) in [np.ndarray,list,tuple]:
				data_list = list(data)
				combined_features.extend(data_list)
				combined_features_sizes[e] = (len(data_list), type(data[0]))
			else:
				combined_features.append(data)
				combined_features_sizes[e] = (1, type(data))
		combined_features_list.append(combined_features)
	#print(combined_features_list[0])

	# Re-sample data
	combined_features_list = np.array(combined_features_list, dtype=np.object)
	#combined_features_list, set['y'] = over_sampling.RandomOverSampler(sampling_strategy='all').fit_sample(combined_features_list, set['y'])
	combined_features_list, set['y'] = resampling_fn().fit_sample(combined_features_list, set['y'])

	# Separate features
	new_combined_features_list = []
	for combined_features in combined_features_list:
		new_combined_features = []
		start = 0
		for e,(size,dtype) in combined_features_sizes.items():
			feature = combined_features[start:start+size]
			if size > 1:
				#feature = np.array(feature, dtype=dtype)
				feature = np.array(feature, dtype=np.float32)
			else:
				feature = feature[0]
			new_combined_features.append(feature)
			start += size
		new_combined_features_list.append(new_combined_features)
	#print(new_combined_features_list[0])
	separated_features = list(zip(*new_combined_features_list))

	for feature, value in zip(set['x'].keys(), separated_features):
		set['x'][feature] = value
	print('Dataset size after re-sampling:', len(set['y']))
	numpyfy_dataset(set)


# Define function for getting the dataframe feature shapes

# In[ ]:


def get_dataframe_feature_shape(df, feature):
	first_element = df[feature][0]
	if type(first_element) not in [np.array,np.ndarray]:
		return None    
	#print(type(first_element), first_element)
	return tf.feature_column.numeric_column(feature, shape=first_element.shape)


# Define function to convert a data-set into a data-list

# In[ ]:


def listify_dataset(dataset):
	dataset_xs = zip(*dataset['x'].values())
	dataset_xs = map(lambda x: tuple((k,v) for k,v in zip(dataset['x'].keys(),x)), dataset_xs)
	return list(zip(dataset_xs, dataset['y']))


# Define function to convert a data-set into a data-list

# In[ ]:


def dictify_datalist(datalist):
	xs, y = zip(*datalist)
	y_list = np.array(y)
	xs = zip(*xs)
	xs_dict = {}
	for x_tuples in xs:
		feature_names, x_tuples = zip(*x_tuples)
		feature = feature_names[0]
		xs_dict[feature] = np.array(x_tuples)
		#print(feature, len(xs_dict[feature]))
	#print('y', len(y_list))
	return {
		'x': xs_dict,
		'y': y_list
	}


# Define the DNN classifier model

# In[ ]:


def build_model_fn(feature_columns, n_classes, config):
	def model_fn(
		features, # This is batch_features from input_fn
		labels,   # This is batch_labels from input_fn
		mode):	# And instance of tf.estimator.ModeKeys, see below

		if mode == tf.estimator.ModeKeys.PREDICT:
			tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
		elif mode == tf.estimator.ModeKeys.EVAL:
			tf.logging.info("my_model_fn: EVAL, {}".format(mode))
		elif mode == tf.estimator.ModeKeys.TRAIN:
			tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

		# Create the layer of input
		input_layer = tf.feature_column.input_layer(features, feature_columns)
		#input_layer = tf.expand_dims(input_layer, 1)

		input_layer = tf.layers.Dense(config['UNITS'], #3, padding='same',
			activation=config['ACTIVATION_FUNCTION'], 
			#kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.003)
		)(input_layer)

		input_layer = tf.layers.Dropout()(input_layer)
		#input_layer = tf.layers.Flatten()(input_layer)

		logits = tf.layers.Dense(n_classes, 
			#kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.003)
		)(input_layer)

		# class_ids will be the model prediction for the class (Iris flower type)
		# The output node with the highest value is our prediction
		def sample(logits, random=True):
			if random:
				u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
				logits -= tf.log(-tf.log(u))
			return tf.argmax(logits, axis=1)

		predictions = { 'class_ids': sample(logits, random=False), 'probabilities': tf.nn.softmax(logits) }

		# 1. Prediction mode
		# Return our prediction
		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode, predictions=predictions)

		# Evaluation and Training mode

		# Calculate the loss
		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		loss += tf.losses.get_regularization_loss()

		# Calculate the accuracy between the true labels, and our predictions
		y_true=labels
		y_pred=predictions['class_ids']
		average_type_list = ['micro','macro','weighted']
		metrics = {}
		for average in average_type_list:
			metrics[f'precision_{average}'] = tf_metrics.precision(y_true, y_pred, n_classes, average=average)
			metrics[f'recall_{average}'] = tf_metrics.recall(y_true, y_pred, n_classes, average=average)
			metrics[f'f1_{average}'] = tf_metrics.f1(y_true, y_pred, n_classes, average=average)

		# 2. Evaluation mode
		# Return our loss (which is used to evaluate our model)
		# Set the TensorBoard scalar my_accurace to the accuracy
		# Obs: This function only sets value during mode == ModeKeys.EVAL
		# To set values during training, see tf.summary.scalar
		if mode == tf.estimator.ModeKeys.EVAL:
			return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

		# If mode is not PREDICT nor EVAL, then we must be in TRAIN
		assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

		# 3. Training mode

		# Default optimizer for DNNClassifier: Adagrad with learning rate=0.05
		# Our objective (train_op) is to minimize loss
		# Provide global step counter (used to count gradient updates)
		#optimizer = tf.train.AdagradOptimizer(0.05)
		#optimizer = tf.train.AdamOptimizer()
		optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=config['LEARNING_RATE'], l2_regularization_strength=config['REGULARIZATION_STRENGTH'])
		train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

		# For Tensorboard
		for metric_name, metric in metrics.items():
			tf.summary.scalar(metric_name, metric[1])

		# Return training operations: loss and train_op
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
	return model_fn


# Define function for training and evaluating

# In[ ]:


def train_and_evaluate(config, trainset, testset, num_epochs, batch_size, max_steps, model_dir, feature_columns, n_classes):
    # Create a custom estimator using model_fn to define the model
    tf.logging.info("Before classifier construction")
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        #save_checkpoints_secs=EVALUATION_SECONDS, 
        save_checkpoints_steps=EVALUATION_STEPS,
        keep_checkpoint_max=1,
    )
    estimator = tf.estimator.Estimator(
        model_fn=build_model_fn(feature_columns, n_classes, config),
        config=run_config,
    )
    tf.logging.info("...done constructing classifier")

    # Build train input callback
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=trainset['x'],
        y=trainset['y'],
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True
    )
    # Build train specifics
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, 
        max_steps=max_steps
    )
    # Build test input callback
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=testset['x'],
        y=testset['y'],
        num_epochs=1,
        batch_size=batch_size,
        shuffle=False
    )
    # Build best_exporter
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    exporter = tf.estimator.BestExporter(
        name="best_exporter",
        serving_input_receiver_fn=tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec),
        exports_to_keep=1 # this will keep the N best checkpoints
    )
    # Build eval specifics
    eval_spec = tf.estimator.EvalSpec(
        input_fn=test_input_fn, 
        steps=EVALUATION_STEPS, 
        start_delay_secs=0, 
        throttle_secs=0,
        exporters=[exporter],
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return estimator


# Define function for extracting summaries (statistics) from tensorboard events

# In[ ]:


def get_summary_results(summary_dir):
	def get_document_list(directory):
		doc_list = []
		for obj in os.listdir(directory):
			obj_path = os.path.join(directory, obj)
			if os.path.isfile(obj_path):
				doc_list.append(obj_path)
			elif os.path.isdir(obj_path):
				doc_list.extend(get_document_list(obj_path))
		return doc_list
    
	def my_summary_iterator(path):
		for r in tf_record.tf_record_iterator(path):
			yield event_pb2.Event.FromString(r)

	result_list = []
	document_list = get_document_list(summary_dir)
	#print(document_list)
	for filename in document_list:
		#print(filename)
		if not os.path.basename(filename).startswith('events.'):
			continue
		value_dict = {}
		for event in my_summary_iterator(filename):
			for value in event.summary.value:
				tag = value.tag
				if tag not in value_dict:
					value_dict[tag]=[]
				value_dict[tag].append((event.step, value.simple_value))
		result_list.append({'event_name':filename, 'results':value_dict})
	return result_list


# Define function for cross-validating the model

# In[ ]:


def cross_validate_model(config, feature_columns, n_classes):
	def get_best_estimator(model_dir):
		exporter_dir = os.path.join(model_dir,'export','best_exporter')
		best_model_name = os.listdir(exporter_dir)[0]
		best_model_path = os.path.join(exporter_dir, best_model_name)
		if not os.path.exists(best_model_path):
			return None
		return tf.estimator.Estimator(
			model_fn=build_model_fn(feature_columns, n_classes, config),
			warm_start_from=best_model_path,
		)

	def save_prediction_results(path, datalist, estimator):
		if estimator is None:
			return
		dataset = dictify_datalist(datalist)
		input_fn = tf.estimator.inputs.numpy_input_fn(x=dataset['x'], y=dataset['y'], shuffle=False)
		dataset['x']['probabilities'] = list(map(lambda x: x['probabilities'], estimator.predict(input_fn)))
		xs,ys = zip(*listify_dataset(dataset))
		xs = [
			{
				k: v.tolist() if type(v) in [np.array,np.ndarray] else v
				for k,v in x
				if k in ['anchorsent','probabilities']
			}
			for x in xs
		]
		with open(path,'w') as f:
			json.dump(xs, f, indent=4)

	# Perform k-fold cross-validation
	feature_set = set(f.key for f in feature_columns)
	cross_validation = KFold(n_splits=config["N_SPLITS"], shuffle=True, random_state=1)
	for e, (train_index, test_index) in enumerate(cross_validation.split(datalist)):
		print(f'-------- Fold {e} --------')
		print(f'Train-set {e} indexes {train_index}')
		print(f'Test-set {e} indexes {test_index}')
		# Split training and test set
		trainlist = [datalist[u] for u in train_index]
		trainset = dictify_datalist(trainlist)
		trainset['x'] = {
			k:v
			for k,v in trainset['x'].items()
			if k in feature_set
		}
		# Re-sample training set (after sentences embedding)
		resample_dataset(trainset, resampling_fn=config["RESAMPLING_FN"])
		print(f'Train-set {e} distribution', Counter(trainset['y']))
		testlist = [datalist[u] for u in test_index]
		testset = dictify_datalist(testlist)
		print(f'Test-set {e} distribution', Counter(testset['y']))

		#config_str = '_'.join(f'{key}={value if not callable(value) else value.__name__}' for key,value in config.items())
		model_dir = f'{MODEL_DIR}{e}'#'-{config_str}'
		estimator = train_and_evaluate(
			config=config,
			trainset=trainset, 
			testset=testset, 
			num_epochs=TRAIN_EPOCHS, 
			batch_size=config["BATCH_SIZE"], 
			max_steps=MAX_STEPS, 
			model_dir=model_dir, 
			feature_columns=feature_columns, 
			n_classes=n_classes
		)

		best_estimator = get_best_estimator(model_dir)
		save_prediction_results(os.path.join(model_dir,'trainset_predictions.json'), trainlist, best_estimator)
		save_prediction_results(os.path.join(model_dir,'testset_predictions.json'), testlist, best_estimator)
        
		yield get_summary_results(os.path.join('.',model_dir,'eval'))[-1]['results'] # iterator


# Define function for distributed cross-validation

# In[ ]:


def ray_cross_validation(datalist,feature_columns,n_classes):
	def get_best_stat_dict(summary_results_list):
		best_stat_dict = {}
		for summary_results in summary_results_list:
			for stat, value_list in summary_results.items():
				_,value_list=zip(*value_list)
				if not re.search(r'(f1|precision|recall)', stat):
					continue
				if stat not in best_stat_dict:
					best_stat_dict[stat] = []
				best_stat_dict[stat].append(np.mean(sorted(value_list, reverse=True)[:3]))
		for stat,best_list in best_stat_dict.items():
			best_stat_dict[stat] = {'mean':np.mean(best_list), 'std':np.std(best_list)}
		return best_stat_dict
            
	def ray_cross_validate_model(config, reporter):
		warnings.filterwarnings('ignore')
		tf.get_logger().setLevel('ERROR')
		summary_results_list = []
		for e,summary_results in enumerate(cross_validate_model(config,feature_columns,n_classes)):
			summary_results_list.append(summary_results)
			print(f'Test-set {e} results:', summary_results)
			best_stat_dict = get_best_stat_dict(summary_results_list)
			reporter(
				timesteps_total=e, 
				# F1 scores
				f1_macro_mean=best_stat_dict["f1_macro"]["mean"],
				f1_macro_std=best_stat_dict["f1_macro"]["std"],
				f1_micro_mean=best_stat_dict["f1_micro"]["mean"],
				f1_micro_std=best_stat_dict["f1_micro"]["std"],
				f1_weighted_mean=best_stat_dict["f1_weighted"]["mean"],
				f1_weighted_std=best_stat_dict["f1_weighted"]["std"],
				# Precision scores
				precision_macro_mean=best_stat_dict["precision_macro"]["mean"],
				precision_macro_std=best_stat_dict["precision_macro"]["std"],
				precision_micro_mean=best_stat_dict["precision_micro"]["mean"],
				precision_micro_std=best_stat_dict["precision_micro"]["std"],
				precision_weighted_mean=best_stat_dict["precision_weighted"]["mean"],
				precision_weighted_std=best_stat_dict["precision_weighted"]["std"],
				# Recall scores
				recall_macro_mean=best_stat_dict["recall_macro"]["mean"],
				recall_macro_std=best_stat_dict["recall_macro"]["std"],
				recall_micro_mean=best_stat_dict["recall_micro"]["mean"],
				recall_micro_std=best_stat_dict["recall_micro"]["std"],
				recall_weighted_mean=best_stat_dict["recall_weighted"]["mean"],
				recall_weighted_std=best_stat_dict["recall_weighted"]["std"],
			)
			print(f'Average best statistics at fold {e}: {best_stat_dict}')
	return ray_cross_validate_model


# Load dataset 1

# In[ ]:


trainset = get_dataframe(os.path.join(DATA_DIR,'training_all'))


# Load dataset 2

# In[ ]:


testset = get_dataframe(os.path.join(DATA_DIR,'test_groundtruth_all'))


# Encode dataset

# In[ ]:


n_classes = encode_dataset({'train':trainset, 'test':testset})


# Get feature columns

# In[ ]:


feature_columns = [
    get_dataframe_feature_shape(trainset['x'],feature) 
    for feature in trainset['x'].keys()
    if get_dataframe_feature_shape(trainset['x'],feature) is not None
    #and feature in ['aclarc_prediction','scicite_prediction']
]
print(feature_columns)


# Merge dataset 1 and 2, because they have different distributions and thus we have to build new train and test sets. Before mergin we convert the datasets into datalists, this way we can easily shuffle them.

# In[ ]:


trainlist = listify_dataset(trainset)
if USE_TEST_SET:
	testlist = listify_dataset(testset)
	datalist = trainlist + testlist
else:
	datalist = trainlist


# Initialize ray

# In[ ]:


ray.init(num_cpus=multiprocessing.cpu_count())


# N.B. Do not use code imported with sys.path.append inside ray distributed code: https://stackoverflow.com/questions/54338013/parallel-import-a-python-file-from-sibling-folder

# Perform automatic hyper-parameters tuning

# In[ ]:


experiment_name = 'hp_tuning'
local_dir = os.path.join('.','ray_results')
analysis = tune.run( # https://ray.readthedocs.io/en/latest/tune-package-ref.html#ray.tune.run
    ray_cross_validation(datalist,feature_columns,n_classes),
    num_samples=1, # Number of times to sample from the hyperparameter space. Defaults to 1. If grid_search is provided as an argument, the grid will be repeated num_samples of times.
    name=experiment_name,
    local_dir=local_dir,
    resume=os.path.isdir(os.path.join(local_dir,experiment_name)),
    #global_checkpoint_period=15*60,
    #keep_checkpoints_num=3,
    verbose=1, # 0, 1, or 2. Verbosity mode. 0 = silent, 1 = only status updates, 2 = status and trial results.
    config={ 
        "N_SPLITS": tune.grid_search([
            #3,
            #4,
            5,
        ]), 
        "RESAMPLING_FN": tune.grid_search([
            None,
            #combine.SMOTEENN, 
            combine.SMOTETomek, 
            #over_sampling.RandomOverSampler,
            over_sampling.SMOTE,
            over_sampling.ADASYN,
            #under_sampling.RandomUnderSampler,
            #under_sampling.EditedNearestNeighbours,
            under_sampling.TomekLinks,
        ]),
        "BATCH_SIZE": tune.grid_search([
            2,
            #3, 
            4,
        ]),
        'UNITS': tune.grid_search([
            4, 
            #6, 
            8, 
            #10,
            12,
        ]),
        'ACTIVATION_FUNCTION': tune.grid_search([
            #None,
            tf.nn.relu,
            #tf.nn.leaky_relu,
            tf.nn.selu,
            tf.nn.tanh,
        ]),
        #'LEARNING_RATE': tune.sample_from(lambda spec: 0.1*3*random.random()),
        'LEARNING_RATE': tune.grid_search([
            0.3,
            0.1,
            0.03,
            0.01,
        ]),
        'REGULARIZATION_STRENGTH': tune.grid_search([
            0.01,
            0.003,
            0.001,
            0.0003,
            0.0001,
        ]),
    },
    scheduler=AsyncHyperBandScheduler(
        metric='f1_macro_mean',
        mode='max',
    )
)


# In[ ]:


#print("Best config: ", analysis.get_best_config(metric='f1_macro_mean'))
analysis_df = analysis.dataframe()
#analysis_df['f1_macro_min'] = analysis_df['f1_macro_mean']-analysis_df['f1_macro_std']
#analysis_df['f1_macro_max'] = analysis_df['f1_macro_mean']+analysis_df['f1_macro_std']
analysis_df['config/RESAMPLING_FN'] = analysis_df['config/RESAMPLING_FN'].map(lambda x: x.split('.')[-1][:-2] if x is not None else x)
best_stats = analysis_df.sort_values(['timesteps_total','f1_macro_mean'], ascending=[False,False]).filter(regex='timesteps_total|macro|config|logdir').iloc[:10]
best_stats.style

