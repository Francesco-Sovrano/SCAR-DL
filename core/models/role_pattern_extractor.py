from models.model_manager import ModelManager
from models.concept_extractor import ConceptExtractor as CE
import re

class RolePatternExtractor(ModelManager):
	FELLOW_IDENTIFIER = [ # https://universaldependencies.org/u/dep/all.html
		'aux',
		'neg',
		'prep',
		'agent',
	]

	FELLOW_REGEXP = re.compile('|'.join(FELLOW_IDENTIFIER))
	
	def __init__(self, model_options):
		super().__init__(model_options.get('tf_model', None))
		self.stringify = model_options.get('stringify', True)

	@staticmethod
	def is_passive(span): # return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
		for token in span:
			if token.dep_ == "auxpass":
				return True
		return False

	@staticmethod
	def get_predicate_core(token):
		if token.pos_ == 'VERB':
			return token
		for token in token.ancestors:
			if token.pos_ == 'VERB':
				return token
		return None

	@staticmethod
	def get_concept_core(span):
		is_valid_dep = lambda x: re.search(CE.CORE_CONCEPT_REGEXP, x)
		for token in span:
			token_dep = CE.get_token_dependency(token)
			if is_valid_dep(token_dep):
				return token
		return None

	def get_predicate(self, token):
		#token = self.get_concept_core(span)
		#if token is None:
		#	return None
		
		predicate_core = self.get_predicate_core(token)
		if predicate_core is None:
			return None

		#for child in predicate_core.children:
		#	print(child, CE.get_token_dependency(child))
		predicate = [
			child
			for child in predicate_core.children
			if re.search(RolePatternExtractor.FELLOW_REGEXP, CE.get_token_dependency(child))
		]
		predicate.append(predicate_core)
		predicate = sorted(predicate, key=lambda x: x.i)#, reverse=False if 'subj' in token.dep_ else True)

		predicate_dict = { 
			'predicate_span': predicate, 
			'predicate_core': predicate_core,
			#'dependency': CE.get_token_dependency(token), 
			#'pid': predicate_core.i,
		}
		if self.stringify:
			predicate_dict['predicate'] = CE.get_concept_text(predicate)
		return predicate_dict

	def get_role_pattern_list(self, text):
		parsed_text = self.nlp(text)
		concept_list = CE.get_concept_list(parsed_text)
		#print(list(concept_list))
		#concept_list = (concept for concept in concept_list if len(concept)==1)
		role_pattern_list = []
		for concept_dict in concept_list:
			core_concept = concept_dict['core']
			predicate_dict = self.get_predicate(core_concept)
			if predicate_dict is None or len(predicate_dict['predicate']) == 0:
				continue
			role_dict = {
				'concept_span': concept_dict['concept'],
				'concept_core': core_concept,
				'dependency': CE.get_token_dependency(core_concept), 
			}
			if self.stringify:
				role_dict['concept'] = CE.get_concept_text(concept_dict['concept'])
			role_dict.update(predicate_dict)
			role_pattern_list.append(role_dict)
		return role_pattern_list