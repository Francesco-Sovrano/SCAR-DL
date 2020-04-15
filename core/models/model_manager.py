import spacy # for natural language processing
# python3 -m spacy download en_core_web_md
from sklearn.preprocessing import normalize
import os
import re
import numpy as np
from misc.doc_reader import get_content_list

SPACY_MODEL = 'en_core_web_md'
MODULE_URL = {
	'USE_Transformer': {
		'local':'/Users/toor/Desktop/NLP/Tutorials/Sentence Embedding/Universal Sentence Encoder/slow', 
		'remote': 'https://tfhub.dev/google/universal-sentence-encoder-large/3',
	},
	'USE_DAN': {
		'local':'/Users/toor/Desktop/NLP/Tutorials/Sentence Embedding/Universal Sentence Encoder/fast',
		'remote': 'https://tfhub.dev/google/universal-sentence-encoder/2',
	},
	'USE_MLQA': {
		'local':'/Users/toor/Desktop/NLP/Tutorials/Sentence Embedding/Universal Sentence Encoder/multilingual-qa',
		'remote': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1',
	},
}

class ModelManager():
	nlp = None
	__tf_placeholders_dict = None
	__session = None
	__docvec_dict = {}
	__tf_model = None
	
	@property
	def docvec_dict(self):
		return ModelManager.__docvec_dict
	
	## the attribute name and the method name must be same which is used to set the value for the attribute
	@docvec_dict.setter
	def docvec_dict(self, var):
		ModelManager.__docvec_dict = var
	
	@staticmethod
	def cache_docvecs(doc_list):
		if ModelManager.__session is not None:
			message_embeddings = ModelManager.embed(doc_list)
			docvec_dict = {doc:vec for doc,vec in zip(doc_list, message_embeddings)}
			ModelManager.__docvec_dict.update(docvec_dict)
	
	@staticmethod
	def load_nlp_model():
		print('Loading Spacy model <{}>...'.format(SPACY_MODEL))
		# go here <https://spacy.io/usage/processing-pipelines> for more information about Language Processing Pipeline (tokenizer, tagger, parser, etc..)
		nlp = spacy.load(SPACY_MODEL)
		print('Spacy model loaded')
		return nlp
	
	@staticmethod
	def load_tf_model(tf_model):
		import tensorflow as tf
		from tensorflow_hub import Module as TFHubModule
		import tf_sentencepiece
		# Reduce logging output.
		#tf.logging.set_verbosity(tf.logging.ERROR)
		
		# Create graph and finalize (finalizing optional but recommended).
		g = tf.Graph()
		with g.as_default():
			# We will be feeding 1D tensors of text into the graph.
			text_input = tf.placeholder(dtype=tf.string, shape=[None])
			model_dict = MODULE_URL[tf_model]
			model_url = model_dict['local'] if os.path.isdir(model_dict['local']) else model_dict['remote']
			embed = TFHubModule(model_url, trainable=False)
			embedded_text = embed(text_input)
			init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
		g.finalize() # Finalizes this graph, making it read-only.
		
		# Create session and initialize.
		session = tf.Session(graph=g, config=tf.ConfigProto(use_per_session_threads=False))
		session.run(init_op)
		tf_placeholders = {
			'embedded_text': embedded_text,
			'text_input': text_input
		}
		return tf_placeholders, session

	@staticmethod
	def cached_embed(queries, norm=None):
		missing_queries = [q for q in queries if q not in ModelManager.__docvec_dict] 
		if len(missing_queries) > 0:
			embeddings = ModelManager.embed(missing_queries, norm)
			ModelManager.__docvec_dict.update({doc:vec for doc,vec in zip(missing_queries, embeddings)})
		query_embeddings = [ModelManager.__docvec_dict[q] for q in queries]
		return query_embeddings
	
	@staticmethod
	def embed(doc_list, norm=None):
		# Feed doc_list into current tf graph
		embedding = ModelManager.__session.run(
			ModelManager.__tf_placeholders_dict['embedded_text'], 
			feed_dict={ModelManager.__tf_placeholders_dict['text_input']: doc_list}
		)
		# Normalize the embeddings, if required
		if norm is not None:
			embedding = normalize(embedding, norm=norm)
		return embedding

	@staticmethod
	def get_similarity_vector(source_text, target_text_list, similarity_fn=np.inner, cached=True):
		embedding_fn = ModelManager.cached_embed if cached else ModelManager.embed
		embeddings = embedding_fn([source_text]+list(target_text_list))
		source_embedding = embeddings[0]
		target_embeddings = embeddings[1:]
		return similarity_fn(source_embedding,target_embeddings)

	@staticmethod
	def find_most_similar(source_text, target_text_list, similarity_fn=np.inner, cached=True):
		similarity_vec = ModelManager.get_similarity_vector(
			source_text=source_text, 
			target_text_list=target_text_list, 
			similarity_fn=similarity_fn, 
			cached=cached,
		)
		argmax = np.argmax(similarity_vec)
		return argmax, similarity_vec[argmax]

	@staticmethod
	def filter_content(content):
		paragraph_list = []
		for text in content.split('\n\n'):
			if text.count(' ') < 3:
				continue
			parsed_text = ModelManager.nlp(text)
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

	@staticmethod
	def get_sentence_iterator_from_docpath(docpath):
		return (
			sentence
			for doc in get_content_list(docpath)
			for sentence in ModelManager.nlp(ModelManager.filter_content(doc)).sents
		)
	
	def __init__(self, tf_model=None):
		# Load Spacy
		if ModelManager.nlp is None:
			ModelManager.nlp = ModelManager.load_nlp_model()
		# Load TF model
		if tf_model is None:
			__tf_placeholders_dict = None
			__session = None
			__docvec_dict = {}
			__tf_model = None
		elif tf_model != ModelManager.__tf_model:
			if ModelManager.__tf_model is None:
				ModelManager.__tf_model = tf_model
				ModelManager.__tf_placeholders_dict, ModelManager.__session = ModelManager.load_tf_model(tf_model)
			else:
				raise ValueError('Cannot load {} in this process, because {} has been already loaded.'.format(tf_model, ModelManager.__tf_model))
