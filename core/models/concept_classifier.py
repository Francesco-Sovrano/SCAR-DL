from models.sentence_classifier import SentenceClassifier
from models.concept_extractor import ConceptExtractor
from misc.onto_reader import get_concept_dict as get_ontology_concept_dict
from more_itertools import unique_everseen

class ConceptClassifier(SentenceClassifier):
	
	def __init__(self, model_options, ontology_path, policy_path):
		self.policy_concept_counter_dict = ConceptExtractor(model_options).get_concept_dict_from_docpath(policy_path)
		self.ontology_concept_dict = get_ontology_concept_dict(ontology_path)

		SentenceClassifier.doc_list = [
			list(unique_everseen(
				(
					(key,description) 
					for key, value in self.ontology_concept_dict.items() 
					for description in value
				),
				key=lambda x: x[1]
			))
		]
		#self.ontology_concept_inverse_dict = dict(map(lambda x: (x[1],x[0]), SentenceClassifier.doc_list))
		super().__init__(0, model_options)
	
	def lemmatize_spacy_document(self, doc):
		return [
			token.lemma_
			for token in doc
			# Remove stop tokens: <https://stackoverflow.com/questions/40288323/what-do-spacys-part-of-speech-and-dependency-tags-mean>
			#if not (token.is_punct or token.pos_ in ['PART','DET','ADP','CONJ','SCONJ'])
		]
	
	def get_concept_dict(self, threshold=0.5):
		concept_dict = {}
		for concept, value_dict in self.policy_concept_counter_dict.items():
			similarity_dict = self.get_query_similarity(concept)
			concept_dict[concept] = {
				'count': value_dict['count'],
				'similar_to': self.get_index_of_most_similar_documents(similarity_dict['weighted'], size=1, threshold=threshold),
			}
		return concept_dict
