import os
import re
import json
from bs4 import BeautifulSoup
from tika import parser

def get_document_list(directory):
	doc_list = []
	for obj in os.listdir(directory):
		obj_path = os.path.join(directory, obj)
		if os.path.isfile(obj_path):
			doc_list.append(obj_path)
		elif os.path.isdir(obj_path):
			doc_list.extend(get_document_list(obj_path))
	return doc_list

def read_html_file(filename):
	file = open(filename, 'r')
	file_content = file.read()
	doc = BeautifulSoup(file_content, features="lxml")
	for script in doc(["script", "style"]): # remove all javascript and stylesheet code
		script.extract()
	return doc.get_text()

def read_pdf_file(pdf_file):
	raw = parser.from_file(pdf_file)
	content = raw['content'].strip()
	content = re.sub(r'([^\x00-\x7F]|-)+\n+', '', content) # remove line-breaks
	content = re.sub(r'[^\x00-\x7F]+','', content) # remove non-ascii chars
	paragraph_list = [
		re.sub(r'( |\n|\t)+', ' ', text.strip())
		for text in content.split('\n\n')
	]
	return '\n\n'.join(paragraph_list)

def get_content_list(directory):
	doc_list = get_document_list(directory)
	content_list = []
	file_name = lambda x: os.path.splitext(x)[0]
	for obj_path in doc_list:
		content = None
		if obj_path.endswith(('.html','.htm')):
			print('Parsing:', obj_path)
			content = read_html_file(obj_path)
		elif obj_path.endswith(('.pdf',)):
			html_file = file_name(obj_path)+'.html'
			if not os.path.isfile(html_file):
				print('Parsing:', obj_path)
				content = read_pdf_file(obj_path)
				with open(html_file, 'w') as f:
					f.write('<html>\n' + content + '\n</html>')
		if content is not None:
			content_list.append(content)
			'''
			with open(file_name(obj_path)+'.json', 'w') as f:
				json.dump(
					[{'sentence': s.text.strip(), 'fred_output': []} for s in NLP(content).sents if s.text.count(' ')>2], 
					f, 
					indent=4,
					ensure_ascii=False,
				)
			'''
	return content_list
