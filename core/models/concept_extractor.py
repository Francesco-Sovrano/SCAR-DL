from models.model_manager import ModelManager
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from more_itertools import unique_everseen
from misc.doc_reader import get_content_list

class ConceptExtractor(ModelManager):
	CONCEPT_IDENTIFIER = [
		'subj',
		'obj'
	]
	FELLOW_IDENTIFIER = [ # https://universaldependencies.org/u/dep/all.html
		'comp', 'mod', # Clauses
		'fixed', 'flat', 'compound',
		'acl', 'pos', 'fixed', 'prep', 'aux'
	]

	CORE_CONCEPT_REGEXP = re.compile('|'.join(CONCEPT_IDENTIFIER))
	COMPOSITE_CONCEPT_REGEXP = re.compile('|'.join(FELLOW_IDENTIFIER))
	CONCEPT_GROUP_REGEXP = re.compile('|'.join(CONCEPT_IDENTIFIER+FELLOW_IDENTIFIER))
	
	def __init__(self, model_options):
		super().__init__(model_options['tf_model'])

	@staticmethod
	def get_concept_text(concept):
		return ' '.join(c.text for c in concept)

	@staticmethod
	def get_fellow_list(token, regexp, trim_prepositions=True):
		children_list = []
		for a in token.children:
			'''
			if a.pos_ == 'VERB':
				continue
			'''
			if re.search(regexp, ConceptExtractor.get_token_dependency(a)):
				children_list.append(a)
				children_list.extend(ConceptExtractor.get_fellow_list(a, regexp, False))
		if trim_prepositions:
			while len(children_list) > 0 and ConceptExtractor.get_token_dependency(children_list[-1]) == 'prep':
				del children_list[-1]
			while len(children_list) > 0 and ConceptExtractor.get_token_dependency(children_list[0]) == 'prep':
				del children_list[0]
		return children_list

	@staticmethod
	def get_token_dependency(token):
		if token.dep_ != 'conj':
			return token.dep_
		for t in token.ancestors:
			if t.dep_ != 'conj':
				return t.dep_

	@staticmethod
	def get_concept_list(processed_doc):
		core_concept_list = [
			token
			for token in processed_doc
			if re.search(ConceptExtractor.CORE_CONCEPT_REGEXP, ConceptExtractor.get_token_dependency(token))
		]
		#print([(token.text,ConceptExtractor.get_token_dependency(token),list(token.ancestors)) for token in core_concept_list])

		concept_list = []
		for t in core_concept_list:
			core_concept = [t]
			concept_group = sorted(core_concept + ConceptExtractor.get_fellow_list(t, ConceptExtractor.CONCEPT_GROUP_REGEXP), key=lambda x: x.idx)
			composite_concept = sorted(core_concept + ConceptExtractor.get_fellow_list(t, ConceptExtractor.COMPOSITE_CONCEPT_REGEXP), key=lambda x: x.idx)

			related_concept_list = [core_concept, concept_group, composite_concept]
			sub_concept_group = []
			for c in concept_group:
				if ConceptExtractor.get_token_dependency(c) != 'prep':
					sub_concept_group.append(c)
				if ConceptExtractor.get_token_dependency(c) == 'prep' or re.search(ConceptExtractor.CORE_CONCEPT_REGEXP, ConceptExtractor.get_token_dependency(c)):
					if len(sub_concept_group) > 0:
						related_concept_list.append(sub_concept_group)
						sub_concept_group = []
			if len(sub_concept_group) > 0:
				related_concept_list.append(sub_concept_group)
			related_concept_list = list(map(lambda x: {'concept':x, 'core':t}, related_concept_list))
			#related_concept_list = unique_everseen(related_concept_list, key=lambda c: ConceptExtractor.get_concept_text(c))
			concept_list.extend(related_concept_list)

		concept_list = unique_everseen(concept_list, key=lambda c: ConceptExtractor.get_concept_text(c['concept']).lower().strip())
		return concept_list

	def get_concept_dict(self, sentence_list):
		concept_dict = {}
		for sentence in sentence_list:
			if sentence.text.count(' ') < 3:
				continue
			sentence_embedding = self.embed([sentence.text])
			sentence_concept_list = [
				ConceptExtractor.get_concept_text(concept_dict['concept']).lower() # do not lemmatize syntagmas
				if len(concept_dict['concept']) > 1 else
				concept_dict['concept'][0].lemma_.lower() # lemmatize words
				for concept_dict in ConceptExtractor.get_concept_list(sentence)
				if len(concept_dict['concept']) > 0
			]
			for concept in sentence_concept_list:
				if concept not in concept_dict:
					concept_dict[concept] = {
						'count': 1, 
						'embedding': self.cached_embed([concept]), 
						'sentence_embedding_list': []
					}
				else:
					concept_dict[concept]['count'] += 1
			for concept in unique_everseen(sentence_concept_list):
				concept_dict[concept]['sentence_embedding_list'].append(sentence_embedding)

		# remove stopwords
		word_list = list(concept_dict.keys())
		for word in word_list:
			if word == '-pron-':
				del concept_dict[word]
			elif word in stopwords.words('english'):
				del concept_dict[word]
			elif re.search(r'\d+',word):
				del concept_dict[word]

		return concept_dict

	def filter_content(self, content):
		paragraph_list = []
		for text in content.split('\n\n'):
			if text.count(' ') < 3:
				continue
			parsed_text = self.nlp(text)
			'''
			verb_list = [token for token in parsed_text if token.pos_=='VERB']
			if len(verb_list) > 0:
				paragraph_list.append(text)
			'''
			for token in parsed_text:
				if token.pos_=='VERB':
					paragraph_list.append(text)
					break
		content = ' '.join(paragraph_list)
		content = re.sub(r'\. ', '.\n\n', content)
		return content

	def get_sentence_list(self, doc):
		return self.nlp(self.filter_content(doc)).sents

	def get_concept_dict_from_docpath(self, docpath):
		sentence_list = (
			sentence
			for doc in get_content_list(docpath)
			for sentence in self.get_sentence_list(doc)
		)
		return self.get_concept_dict(sentence_list)
	