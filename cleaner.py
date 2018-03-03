from nltk.stem.wordnet import WordNetLemmatizer 
import operator
import string
import nltk 
import csv 
import pandas as pd 
import sys, re

csv.field_size_limit(sys.maxsize)
punctuation = string.punctuation
lem = WordNetLemmatizer()

you_words = ["you", "your", "yours", "yourself", "ur", "u", "ya", "you're", "u'r" "u r"]
url_tags = ["http", "www","0","1","2","3","4","5","6","7","8","9","wp:","see:","user:"]
ignore_stopwords = ["not", "no", "dont"]
inside_ignores = ["jpg"]
yourslist = ["you are", "u r", "you r", "you sound like", "you're", "u'r"]
supportwords = open('data/supportwords.txt').read().strip().split("\n")
badwords_d = open('data/updated_bad_words.txt').read().strip().split("\n")
badwords_l = open('data/bad_words.txt').read().strip().split("\n")
stopwords = open('data/stopwords.txt').read().strip().split("\n")

negatives = open('data/negatives.txt').read().strip().split("\n")
stopwords = [x.replace("\r","") for x in stopwords]
negatives = [x.replace("\r","") for x in negatives]
negatives_d = {}
for n in negatives:
	negatives_d[n.lower()] = 1
badwords_l_d = {}
badwords_ph = []
for n in badwords_l:
	if " " in n.strip():
		badwords_ph.append(n)
	else:
		badwords_l_d[n.lower()] = 1

def _bad_word_cases(txt):	
	for bw in badwords_d:
		bw = bw.split(",")
		wrdx = bw[0]

		wrvx = bw[0].replace("*","")
		if len(bw) > 1:
			wrvx = bw[1]	

		if " "+wrdx+" " in txt:
			txt = txt.replace(" "+wrdx+" ", " "+wrvx+" ")
		elif "*" in bw and bw in txt:
			txt = re.sub(wrdx, wrvx, txt)
	return txt

def _clean(txt):
	# fix decoding 
	txt = txt.decode('utf8').encode('ascii','ignore')

	# lowercasing
	txt = txt.lower()

	txt = " "+txt+" "
	# if question mark in sentence, add QQQ in last
	# if "??" in txt:
	# 	txt = txt + " QQQQ"
	if "?" in txt:
		txt = txt + " QQQ"

	txt = txt.replace(" u "," you ") 
	txt = txt.replace(" em "," them ") 
	txt = txt.replace(" da "," the ") 
	txt = txt.replace(" yo "," you ") 
	txt = txt.replace(" ur "," you ")     
	txt = txt.replace("won't", "will not") 
	txt = txt.replace("can't", "cannot") 
	txt = txt.replace("i'm", "i am") 
	txt = txt.replace(" im ", " i am ") 
	txt = txt.replace("ain't", "is not") 
	txt = txt.replace("'ll", " will") 
	txt = txt.replace("'t", " not") 
	txt = txt.replace("'ve", " have") 
	txt = txt.replace("'s", " is") 
	txt = txt.replace("'re", " are") 
	txt = txt.replace("'d", " would") 

	# replace you words - all are same
	for youword in you_words:
		txt = txt.replace(" "+youword+" "," YYY ")

	# special cases on txt
	txt = _bad_word_cases(txt)

	# process word by word
	words = []
	for wrd in txt.split():	
		if wrd.startswith("fuk"):
			wrd = "fuck"
		elif wrd.startswith("fuc"):
			wrd = "fuck"

		# remove urls
		if any(wrd.startswith(urlx) for urlx in url_tags):
			wrd = ""

		# remove words such as jpg
		if any(insw in wrd for insw in inside_ignores):
			wrd = ""

		# remove supportwords
		# if wrd in supportwords:
		# 	wrd = ""

		# remove punctuations
		for punc in punctuation:
			wrd = wrd.replace(punc, "")

		# lemmatize
		# wrd = lem.lemmatize(wrd, "v")
		# wrd = lem.lemmatize(wrd, "n")

		# remove stopwords
		# if wrd in stopwords and wrd not in ignore_stopwords:
			# wrd = ""

		# remove digits
		if wrd and any(str(wrd)[0] == dig  for dig in "1234567890"):
			wrd = "dig"

		# remove small words
		# if len(wrd.strip()) <= 2:
		# 	continue

		words.append(wrd)

	# cleaned word
	txt = " ".join(words)
	txt = _bad_word_cases(txt)
	txt = txt.strip()
	return txt 

def process_cleaning(file_name, limit = False):
	updated_rows = []
	with open("input/"+file_name+".csv") as data:
		reader = csv.reader(data)
		ind = 0
		for row in reader:
			ind += 1 
			print ind
			if ind == 1:
				continue

			if limit and ind == limit:
				break

			text = row[1]
			cleaned =_clean(text)

			new_row = {}
			labels = ["id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"]
			if file_name == 'test':
				labels = ["id","comment_text"]

			for j, label in enumerate(labels):
				new_row[label] = row[j]
			new_row['comment_text'] = str(cleaned)
			updated_rows.append(new_row)

	out = pd.DataFrame(updated_rows)
	out.to_csv('input/cleaned_'+file_name+'.csv', index = False)

def check_wrd_cnt(txt, lst, d = False):
	if d == True:
		cnt = 0
		for wrd in txt.lower().split():
			wrd_c = "".join(x for x in wrd if x not in punctuation)
			if wrd in lst:
				cnt += 1 
			elif wrd_c in lst:
				cnt += 1 	
	else:
		cnt = 0
		for w in lst:
			if " "+w.lower()+" " in " "+txt.lower()+" ":
				cnt += 1 
	return cnt

def check_badwrd_cnt(txt):
	cnt = 0
	for wrd in str(txt).split():
		if wrd in badwords_l_d:
			cnt += 1

	for badph in badwords_ph:
		if badph.lower() in str(txt).lower():
			cnt += 1 
	return cnt 


def add_features(file_name):
	df = pd.read_csv('data/'+file_name+'.csv')
	print "1"
	df['word_count'] = df['comment_text'].apply(lambda x : len(str(x).split()))
	print "2"
	df['uniq_word_count'] = df['comment_text'].apply(lambda x : len(list(set(str(x).split()))))
	print "3"
	df['char_count'] = df['comment_text'].apply(lambda x : len(str(x).replace(" ","")))
	print "4"
	df['charsp_count'] = df['comment_text'].apply(lambda x : len(str(x)))
	print "5"
	df['you_phrases'] = df['comment_text'].apply(lambda x : check_wrd_cnt(str(x), yourslist))
	print "6"
	df['negative_word_count'] = df['comment_text'].apply(lambda x : check_wrd_cnt(str(x), negatives_d, d = True))
	print "7"
	df['bad_word_count'] = df['comment_text'].apply(lambda x : check_badwrd_cnt(x))
	df.to_csv('data/feats_'+file_name+".csv", index = False)
	return df 

if __name__ == '__main__':
	# file_name = 'train'
	# process_cleaning(file_name)
	# add_features(file_name)

	file_name = 'test'
	# add_features(file_name)
	# process_cleaning(file_name)



# ideas to try 


# Feature Engineering
	# use xgboost to get some features <- no improvement 
	# use 4 grams
	# Changing architecture of Ngrams
		# only ngram model -> f1
		# only char model -> f2
		# ngram+char model -> f3

		# only ngram model -> r1 lr
		# only char model -> r2 lr
		# ngram+char model -> r3
	# elasticnet http://blog.kaggle.com/2014/08/01/learning-from-the-best/
# Dim Redn << No
	# PCA
	# SVD <- not training on local
	# Chisq 
# Ensemble 
	# multiple seeds <- no improvement
# Models 
	# using Logit with SGD
	# using xgboosting
	# use deep learning models - lstm, cnns, word2vec
# Fine Tuning of parameters 
	# ridge params

# https://nycdatascience.com/blog/student-works/improving-model-accuracy-kaggle-competition/