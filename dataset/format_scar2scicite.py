#!/usr/bin/env python
# coding: utf-8

# Install dependencies

# !pip3 install -r requirements.txt -U
# !python3 -m spacy download en_core_web_md

import numpy as np
import pandas as pd
import re
import json

SCICITE_LABELS = [
	"method",
	"background",
	"result",
]

ZERO_CLASS = 'none'
LABELS_TO_EXCLUDE = [
	#'cites',
	'cites_as_review',
	#'extends', 
	#'uses_data_from', 
	#'uses_method_in',
]

def get_dataframe(dataset_file):
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
	
	# Remove null values
	df['anchorsent'].replace(np.NaN, '', inplace=True)
	df['sectype'].replace(np.NaN, 'none', inplace=True)

	df = df[df.anchorsent != '']
	df['anchorsent'] = df['anchorsent'].map(lambda x: re.sub(r'\[\[.*\]\]','',x))
	df['anchorsent'] = df['anchorsent'].map(lambda x: re.sub(r'[^\x00-\x7F]+',' ',x))
	df['anchorsent'] = df['anchorsent'].map(lambda x: re.sub(r"^'(.*)'$",r'\1',x))

	# Print dataframe
	print('Dataframe')
	print(df)

	df = df.rename(columns={
		'anchorsent': 'string', 
		'sectype': 'sectionName', 
		'citfunc': 'label',
		'refid': 'citedPaperId',
		'art': 'citingPaperId',
	})
	# Extract features from dataframe
	df = df[['string','sectionName','label','citedPaperId','citingPaperId']]
	df['source'] = 'explicit'
	df['excerpt_index'] = 0
	df['citedPaperId'] = df['citedPaperId'].map(lambda x: re.sub(r"^<(.*)>$",r'\1',x))
	df['id'] = df['unique_id'] = df["citingPaperId"].map(str) + '>' + df["citedPaperId"].map(str)
	df['label'] = SCICITE_LABELS[-1]
	
	return df.to_dict('records')

def df2jsonl(filename):
	data = get_dataframe(filename)
	with open(filename+'.jsonl','w') as f:
		for entry in data:
			json.dump(entry, f)#, indent=4, sort_keys=True)
			f.write('\n')

df2jsonl('training_all.csv')
df2jsonl('test_groundtruth_all.csv')