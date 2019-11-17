from models.model_manager import ModelManager
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from more_itertools import unique_everseen

class ConceptExtractor(ModelManager):
	CONCEPT_IDENTIFIER = [
		'subj',
		'obj'
	]
	FELLOW_IDENTIFIER = [ # https://universaldependencies.org/u/dep/all.html
		'comp', 'mod', # Clauses
		'fixed', 'flat', #'compound',
		'acl', 'pos', 'prep', 'aux',
		'case', #'agent',
		'neg',
	]

	CORE_CONCEPT_REGEXP = re.compile('|'.join(CONCEPT_IDENTIFIER))
	COMPOSITE_CONCEPT_REGEXP = re.compile('|'.join(FELLOW_IDENTIFIER))
	CONCEPT_GROUP_REGEXP = re.compile('|'.join(CONCEPT_IDENTIFIER+FELLOW_IDENTIFIER))
	
	def __init__(self, model_options):
		self.min_concept_size = model_options.get('min_concept_size',2)
		super().__init__(model_options['tf_model'])

	@staticmethod
	def lemmatize_span(span, prevent_verb_lemmatization=True):
		return ' '.join(
			e.lemma_ if not prevent_verb_lemmatization or e.pos_!='VERB' else e.text 
			for e in span 
			if e.lemma_ != '-PRON-'
		)

	@staticmethod
	def get_concept_text(concept):
		return ' '.join(c.text for c in concept)

	@staticmethod
	def get_fellow_list(token, regexp, min_concept_size=1):
		token_list = []
		for a in token.children:
			if len(a.text) < min_concept_size:
				continue
			'''
			if a.pos_ == 'VERB':
				continue
			'''
			if re.search(regexp, ConceptExtractor.get_token_dependency(a)):
				token_list.append(a)
				token_list.extend(ConceptExtractor.get_fellow_list(a, regexp, min_concept_size))
		return token_list

	@staticmethod
	def trim_prepositions(token_list):
		while len(token_list) > 0 and ConceptExtractor.get_token_dependency(token_list[-1]) == 'prep':
			del token_list[-1]
		while len(token_list) > 0 and ConceptExtractor.get_token_dependency(token_list[0]) == 'prep':
			del token_list[0]
		return token_list

	@staticmethod
	def get_token_dependency(token):
		if token.dep_ != 'conj':
			return token.dep_
		for t in token.ancestors:
			if t.dep_ != 'conj':
				return t.dep_

	@staticmethod
	def get_consecutive_tokens(core_concept, concept_span):
		core_concept_index_in_list = concept_span.index(core_concept)
		core_concept_index_in_span = core_concept.i
		return [
			t
			for i,t in enumerate(concept_span)
			if abs(t.i-core_concept_index_in_span) == abs(i-core_concept_index_in_list)
		]

	@staticmethod
	def get_composite_concept(core_concept, min_concept_size=1):
		concept_span = [core_concept] + ConceptExtractor.get_fellow_list(core_concept, ConceptExtractor.COMPOSITE_CONCEPT_REGEXP, min_concept_size)
		concept_span.sort(key=lambda x: x.idx)
		concept_span = ConceptExtractor.get_consecutive_tokens(core_concept, concept_span)
		concept_span = ConceptExtractor.trim_prepositions(concept_span)
		return concept_span

	@staticmethod
	def get_concept_group(core_concept, min_concept_size=1):
		concept_span = [core_concept] + ConceptExtractor.get_fellow_list(core_concept, ConceptExtractor.CONCEPT_GROUP_REGEXP, min_concept_size)
		concept_span.sort(key=lambda x: x.idx)
		concept_span = ConceptExtractor.get_consecutive_tokens(core_concept, concept_span)
		concept_span = ConceptExtractor.trim_prepositions(concept_span)
		return concept_span

	@staticmethod
	def get_concept_list(processed_doc, min_concept_size=1):
		core_concept_list = [
			token
			#for sentence in processed_doc.sents
			#for token in sentence
			for token in processed_doc
			if re.search(ConceptExtractor.CORE_CONCEPT_REGEXP, ConceptExtractor.get_token_dependency(token))
		]
		#print([(token.text,ConceptExtractor.get_token_dependency(token),list(token.ancestors)) for token in core_concept_list])

		concept_list = []
		for t in core_concept_list:
			if len(t.text) < min_concept_size:
				continue
			core_concept = [t]
			concept_group = ConceptExtractor.get_concept_group(t, min_concept_size)
			composite_concept = ConceptExtractor.get_composite_concept(t, min_concept_size)

			related_concept_list = [core_concept, concept_group, composite_concept]
			sub_concept_group = []
			for c in concept_group:
				c_dep = ConceptExtractor.get_token_dependency(c)
				if c_dep != 'prep':
					sub_concept_group.append(c)
				if c_dep == 'prep' or re.search(ConceptExtractor.CORE_CONCEPT_REGEXP, c_dep):
					if len(sub_concept_group) > 0:
						related_concept_list.append(sub_concept_group)
						sub_concept_group = []
			if len(sub_concept_group) > 0:
				related_concept_list.append(sub_concept_group)
			related_concept_list = list(map(lambda x: {'concept':x, 'core':t}, related_concept_list))
			#related_concept_list = unique_everseen(related_concept_list, key=lambda c: ConceptExtractor.get_concept_text(c))
			concept_list.extend(related_concept_list)

		concept_list = unique_everseen(concept_list, key=lambda c: ConceptExtractor.get_concept_text(c['concept']).lower().strip() + '.' + str(c['core'].i))
		return concept_list

	def build_concept_counter_dict(self, concept_list, concept_dict={}):
		concept_counter = Counter(concept_list)
		concept_dict.update({
			concept: {
				'count': count, 
				'embedding': self.cached_embed([concept]), 
			}
			for concept,count in concept_counter.items()
		})
		return concept_dict

	def get_concept_dict(self, sentence_iterator, remove_stopwords=True):
		concept_dict = {}
		for sentence in sentence_iterator:
			if sentence.text.count(' ') < 3:
				continue
			sentence_embedding = self.embed([sentence.text])
			sentence_concept_list = [
				self.lemmatize_span(concept_dict['concept']).lower() # do lemmatize syntagmas
				for concept_dict in ConceptExtractor.get_concept_list(sentence, self.min_concept_size)
				if len(concept_dict['concept']) > 0
			]
			concept_dict = self.build_concept_counter_dict(sentence_concept_list, concept_dict)
			for concept in unique_everseen(sentence_concept_list):
				if 'sentence_embedding_list' not in concept_dict[concept]:
					concept_dict[concept]['sentence_embedding_list'] = []
				concept_dict[concept]['sentence_embedding_list'].append(sentence_embedding)

		# remove stopwords
		if remove_stopwords:
			word_list = list(concept_dict.keys())
			for word in word_list:
				if word == '':
					del concept_dict[word]
				elif word in stopwords.words('english'):
					del concept_dict[word]
				#elif re.search(r'\d+',word):
				#	del concept_dict[word]

		return concept_dict

	def get_concept_dict_from_docpath(self, docpath):
		sentence_iterator = self.get_sentence_iterator_from_docpath(docpath)
		return self.get_concept_dict(sentence_iterator)
	