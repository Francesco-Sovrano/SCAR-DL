from models.model_manager import ModelManager
from models.concept_extractor import ConceptExtractor as CE
import re

class RolePatternExtractor(ModelManager):
	FELLOW_IDENTIFIER = [ # https://universaldependencies.org/u/dep/all.html
		'aux',
		'neg',
		'prep',
	]

	FELLOW_REGEXP = re.compile('|'.join(FELLOW_IDENTIFIER))
	
	def __init__(self, model_options):
		super().__init__(model_options['tf_model'])
		self.use_lemma = model_options.get('use_lemma',False)

	@staticmethod
	def get_predicate_core(token):
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

	def get_predicate(self, span):
		token = self.get_concept_core(span)
		if token is None:
			return None
		
		predicate_core = self.get_predicate_core(token)
		if predicate_core is None:
			return None

		if not self.use_lemma:
			#for child in predicate_core.children:
			#	print(child, CE.get_token_dependency(child))
			predicate = [
				child
				for child in predicate_core.children
				if re.search(RolePatternExtractor.FELLOW_REGEXP, CE.get_token_dependency(child))
			]
			predicate.append(predicate_core)
			predicate = sorted(predicate, key=lambda x: x.i)#, reverse=False if 'subj' in token.dep_ else True)
			predicate = CE.get_concept_text(predicate)
		else:
			predicate = predicate_core.lemma_
		return {'predicate':predicate, 'dependency':CE.get_token_dependency(token), 'pid': predicate_core.i}

	def get_role_pattern_list(self, text):
		parsed_text = self.nlp(text)
		concept_list = CE.get_concept_list(parsed_text)
		#concept_list = (concept for concept in concept_list if len(concept)==1)
		role_pattern_list = []
		for concept_dict in concept_list:
			predicate_dict = self.get_predicate(concept_dict['concept'])
			if predicate_dict is None or len(predicate_dict['predicate']) == 0:
				continue
			role_dict = {
				'concept': CE.get_concept_text(concept_dict['concept']),
				'core': concept_dict['core'],
			}
			role_dict.update(predicate_dict)
			role_pattern_list.append(predicate_dict)
		return role_pattern_list