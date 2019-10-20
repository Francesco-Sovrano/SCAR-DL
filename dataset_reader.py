import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import re
import os
import tf_metrics
from imblearn import over_sampling, under_sampling, combine
from collections import Counter
#import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from models.role_pattern_extractor import RolePatternExtractor

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Get inputs
import sys
_, trainset_file, testset_file = sys.argv

# Define constants
BATCH_SIZE = 2
N_SPLITS = 5
USE_TEST_SET = False
USE_PATTERN_EMBEDDING = True

ZERO_CLASS = 'none'
LABELS_TO_EXCLUDE = [
	#'cites',
	'cites_as_review',
	#'extends', 
	#'uses_data_from', 
	#'uses_method_in',
]
OVERSAMPLE = False
UNDERSAMPLE = False

TRAIN_EPOCHS = None
EVALUATION_SECONDS = 5
MAX_STEPS = 10**5
MODEL_DIR = './model'
TF_MODEL = 'USE_MLQA'
MODEL_MANAGER = RolePatternExtractor({'tf_model':TF_MODEL})

# Build functions
def get_formatted_dataset(dataset_file):
	# Load dataset
	df = pd.read_csv(dataset_file, sep='	')
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
	# Extract target list
	target_list = df.pop('citfunc').values.tolist()

	# Extract features from dataframe
	df = df[['anchorsent','sectype']]
	
	# Remove null values
	df['anchorsent'].replace(np.NaN, '', inplace=True)
	df['sectype'].replace(np.NaN, 'none', inplace=True)

	df = df[df.anchorsent != '']
	df['anchorsent'] = df['anchorsent'].map(lambda x: re.sub(r'\[\[.*\]\]','',x))
	df['anchorsent'] = df['anchorsent'].map(lambda x: re.sub(r'[^\x00-\x7F]+',' ',x))

	if USE_PATTERN_EMBEDDING:
		extra_list = []
		for text in df['anchorsent'].values:
			extra = list(Counter(pattern['predicate'] for pattern in MODEL_MANAGER.get_role_pattern_list(text)).keys())
			extra_list.append(extra[0] if len(extra)>0 else '')
		df['main_predicate'] = extra_list
	
	# Print dataframe
	print('Dataframe')
	print(df)
	
	# Return dataset
	feature_list = df.columns.values.tolist()
	x_dict = {feature: df[feature].tolist() for feature in feature_list}
	y_list = target_list
	return {'x':x_dict, 'y':y_list}

def numpyfy_dataset(set):
	set['x'] = {k: np.array(v) for k,v in set['x'].items()}
	set['y'] = np.array(set['y'])

def clean_dataset(dataset):
	# Embed anchor sentences into vectors
	for key,value in dataset.items():
		df = value['x']
		# Embed anchor sentences
		cache_file = f'{TF_MODEL}.{key}.anchorsent.embedding_cache.pkl'
		if os.path.isfile(cache_file):
			with open(cache_file, 'rb') as f:
				embedded_sentences = pickle.load(f)
		else:
			df['anchorsent'] = list(df['anchorsent'])
			embedded_sentences = MODEL_MANAGER.embed(df['anchorsent'])
			with open(cache_file, 'wb') as f:
				pickle.dump(embedded_sentences, f)
		df['anchorsent'] = embedded_sentences
		# Embed extra info
		if USE_PATTERN_EMBEDDING:
			cache_file = f'{TF_MODEL}.{key}.extra.embedding_cache.pkl'
			if os.path.isfile(cache_file):
				with open(cache_file, 'rb') as f:
					embedded_extra = pickle.load(f)
			else:
				df['main_predicate'] = list(df['main_predicate'])
				embedded_extra = MODEL_MANAGER.embed(df['main_predicate'])
				with open(cache_file, 'wb') as f:
					pickle.dump(embedded_extra, f)
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
	print('Sectype classes:', list(label_encoder_sectype.classes_))
	for set in dataset.values():
		labeled_sectypes = label_encoder_sectype.transform(set['x']['sectype'])
		set['x']['sectype'] = onehot_encoder_sectype.transform(labeled_sectypes.reshape(-1, 1)).toarray()[:,1:]

	# Input features to numpy array
	for set in dataset.values():
		numpyfy_dataset(set)
	# Return number of target classes
	return len(label_encoder_target.classes_)

def resample_dataset(set):
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

	# Oversample data
	combined_features_list = np.array(combined_features_list, dtype=np.object)
	#combined_features_list, set['y'] = over_sampling.RandomOverSampler(sampling_strategy='all').fit_sample(combined_features_list, set['y'])
	if OVERSAMPLE and UNDERSAMPLE:
		combined_features_list, set['y'] = combine.SMOTETomek().fit_sample(combined_features_list, set['y'])
	elif OVERSAMPLE:
		combined_features_list, set['y'] = over_sampling.ADASYN().fit_sample(combined_features_list, set['y'])
	elif UNDERSAMPLE:
		combined_features_list, set['y'] = under_sampling.NeighbourhoodCleaningRule().fit_sample(combined_features_list, set['y'])

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

def get_dataframe_feature_shape(df, feature):
	first_element = df[feature][0]
	shape = first_element.shape if type(first_element) is np.ndarray else ()
	return tf.feature_column.numeric_column(feature, shape=shape)

def listify_dataset(dataset):
	dataset_xs = zip(*dataset['x'].values())
	dataset_xs = map(lambda x: tuple((k,v) for k,v in zip(dataset['x'].keys(),x)), dataset_xs)
	return list(zip(dataset_xs, dataset['y']))

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

def build_model_fn(feature_columns, n_classes):
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

		input_layer = tf.layers.Dense(16, #3, padding='same',
			activation=tf.nn.tanh, 
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

		predictions = { 'class_ids': sample(logits, random=False) }

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
		optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01, l2_regularization_strength=0.003)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

		# For Tensorboard
		for metric_name, metric in metrics.items():
			tf.summary.scalar(metric_name, metric[1])

		# Return training operations: loss and train_op
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
	return model_fn

def train_and_evaluate(trainset, testset, num_epochs, batch_size, max_steps, model_dir, feature_columns, n_classes):
	# Create a custom estimator using model_fn to define the model
	tf.logging.info("Before classifier construction")
	run_config = tf.estimator.RunConfig(
		model_dir=model_dir,
		save_checkpoints_secs=EVALUATION_SECONDS, 
		#keep_checkpoint_max=3,
	)
	estimator = tf.estimator.Estimator(
		model_fn=build_model_fn(feature_columns, n_classes),
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
	# Build test input callback
	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x=testset['x'],
		y=testset['y'],
		num_epochs=1,
		batch_size=batch_size,
		shuffle=False
	)

	train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps)
	eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, throttle_secs=EVALUATION_SECONDS)

	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# Load dataset
trainset = get_formatted_dataset(trainset_file)
testset = get_formatted_dataset(testset_file)

# Clean dataset
n_classes = clean_dataset({'train':trainset, 'test':testset})

# Get feature columns
feature_columns = [get_dataframe_feature_shape(trainset['x'],feature) for feature in trainset['x'].keys()]

# Get dataset
trainlist = listify_dataset(trainset)
if USE_TEST_SET:
	testlist = listify_dataset(testset)
	datalist = trainlist + testlist
else:
	datalist = trainlist
#random.shuffle(datalist)

# Cross-validate the model
cross_validation = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1)
for e, (train_index, test_index) in enumerate(cross_validation.split(datalist)):
	print(f'-------- Fold {e} --------')
	print(f'Train-set {e} indexes {train_index}')
	print(f'Test-set {e} indexes {test_index}')
	# Split training and test set
	trainlist = [datalist[u] for u in train_index]
	trainset = dictify_datalist(trainlist)
	# Oversample training set (after sentences embedding)
	if OVERSAMPLE or UNDERSAMPLE:
		resample_dataset(trainset)
	print(f'Train-set {e} distribution', Counter(trainset['y']))
	testlist = [datalist[u] for u in test_index]
	testset = dictify_datalist(testlist)
	print(f'Test-set {e} distribution', Counter(testset['y']))

	train_and_evaluate(
		trainset=trainset, 
		testset=testset, 
		num_epochs=TRAIN_EPOCHS, 
		batch_size=BATCH_SIZE, 
		max_steps=MAX_STEPS, 
		model_dir=MODEL_DIR+str(e), 
		feature_columns=feature_columns, 
		n_classes=n_classes
	)
	#break
