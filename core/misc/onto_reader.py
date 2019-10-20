import pandas as pd
from misc.doc_reader import get_document_list
import os
import re
	
def explode_concept_key(key):
	key = re.sub(r"(_|-)", r" ", key)
	key = key.split(':')[-1]
	key = key[0].upper() + key[1:]
	splitted_key = re.findall('[A-Z][^A-Z]*', key)

	# join upper case letters
	i = 0
	j = 1
	while j < len(splitted_key):
		if len(splitted_key[j]) == 1:
			splitted_key[i] += splitted_key[j]
			splitted_key[j] = ''
			j += 1
		else:
			i = j
			j = i+1
	
	exploded_key = ' '.join(splitted_key)
	exploded_key = re.sub(r" +", r" ", exploded_key)
	return exploded_key

def get_dataframe_dict(ontology_dir):
	doc_list = get_document_list(ontology_dir)
	dataframe_dict = {}
	for obj_path in doc_list:
		if obj_path.endswith(('.csv',)):
			print('Parsing:', obj_path)
			_, filename = os.path.split(obj_path)
			class_name = filename.split('.')[0]
			dataframe_dict[class_name] = pd.read_csv(obj_path, sep=';')
	return dataframe_dict

def get_concept_dict(ontology_dir):
	dataframe_dict = get_dataframe_dict(ontology_dir)
	
	concept_dict = {}
	for concept, df in dataframe_dict.items():
		concept_dict[concept] = [explode_concept_key(concept).lower().strip()]
		sub_classes = df['SubClasses'].values.tolist()
		concept_dict.update({
			sc: [explode_concept_key(sc).lower().strip()]
			for sc in sub_classes
		})
	return concept_dict

'''
import sys
_, ontology_path, skos_path = sys.argv

print(get_concept_dict(ontology_path, skos_path))
'''