import numpy as np 
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
import scipy.sparse as sparse
import pickle
from sklearn.grid_search import GridSearchCV

ignores = ["sep", "jan", "feb", "mar", "oct", "nov", "apr", "may", "jun", "jul", "aug", "dec", "utc"]
you_words = ["you", "your", "yours", "yourself", "ur", "u", "ya", "you're", "u'r" "u r"]
url_tags = ["http", "www","0","1","2","3","4","5","6","7","8","9","wp:","see:","user:"]
ignore_stopwords = ["not", "no", "dont"]
inside_ignores = ["jpg"]
yourslist = ["you are", "u r", "you r", "you sound like", "you're", "u'r"]
supportwords = open('data/supportwords.txt').read().strip().split("\n")
badwords_d = open('data/updated_bad_words.txt').read().strip().split("\n")

def JoinAndSanitize(cmt, annot, san = True):
	df = cmt.set_index('rev_id').join(annot.groupby(['rev_id']).mean())
	if san:
		comment = 'comment' if 'comment' in df else 'comment_text'
		df[comment] = df[comment].fillna('erikov')
		df[comment] = df[comment].apply(lambda x : Sanitize(x))
	return df

def isNoise(word):
	noise = False
	if word.startswith("www") or word.startswith("http"):
		noise = True
	elif (any(x == word[0] for x in "0987654321")):
		noise = True
	elif any(word in x for x in ignores):
		noise = True
	return noise 

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

def Sanitize(txt):
	txt = str(txt).replace("NEWLINE_TOKEN", " ")
	txt = str(txt).lower().replace('newline_token', ' ')
	txt = txt.decode('utf8').encode('ascii','ignore')
	txt = " "+txt+" "
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

	for youword in you_words:
		txt = txt.replace(" "+youword+" "," YYY ")
	txt = _bad_word_cases(txt)
	words = []
	for wrd in txt.split():	
		if wrd.startswith("fuk"):
			wrd = "fuck"
		elif wrd.startswith("fuc"):
			wrd = "fuck"
		if any(wrd.startswith(urlx) for urlx in url_tags):
			wrd = ""
		if any(insw in wrd for insw in inside_ignores):
			wrd = ""
		for punc in punctuation:
			wrd = wrd.replace(punc, "")
		if wrd and any(str(wrd)[0] == dig  for dig in "1234567890"):
			wrd = "dig"
		words.append(wrd)
	txt = " ".join(words)
	txt = _bad_word_cases(txt)
	txt = txt.strip()
	return txt 

def Tfidfize(df, name):
	comment = 'comment' if 'comment' in df else 'comment_text'

	print "word", name
	tfidfer = TfidfVectorizer(ngram_range=(1,3), max_features=100000,
				   use_idf=1, stop_words='english', analyzer='word',
				   smooth_idf=1, sublinear_tf=1 )
	tfidf = tfidfer.fit_transform(df[comment])

	sparse.save_npz('data/convai/X_'+name+'_cln.npz', tfidf)
	with open('data/convai/tfidfer_'+name+'_cln.pk', 'wb') as fin:
		pickle.dump(tfidfer, fin)
	

	print "char", name
	tfidfer_char = TfidfVectorizer(ngram_range=(1,3), max_features=40000,
				   use_idf=1, stop_words='english', analyzer='char',
				   smooth_idf=1, sublinear_tf=1 )
	tfidf_char = tfidfer_char.fit_transform(df[comment])

	sparse.save_npz('data/convai/X_'+name+'_char_cln.npz', tfidf_char)
	with open('data/convai/tfidfer_'+name+'_char_cln.pk', 'wb') as fin1:
		pickle.dump(tfidfer_char, fin1)

	print "Completed", name


	print "char", name
	tfidfer_char = TfidfVectorizer(ngram_range=(4,4), max_features=8000,
				   use_idf=1, stop_words='english', analyzer='char',
				   smooth_idf=1, sublinear_tf=1 )
	tfidf_char = tfidfer_char.fit_transform(df[comment])

	sparse.save_npz('data/convai/X_'+name+'_char_cln4.npz', tfidf_char)
	with open('data/convai/tfidfer_'+name+'_char_cln4.pk', 'wb') as fin1:
		pickle.dump(tfidfer_char, fin1)

	print "Completed", name



toxp = 0
aggp = 0
attp = 0

san = False
save = 0
load = 0
predicting = 1

print "reading Y values"
if toxp == 1:
	print "Reading Data Files"
	toxic_cmt = pd.read_table('convai/toxicity_annotated_comments.tsv')
	toxic_annot = pd.read_table('convai/toxicity_annotations.tsv')
	print "Joining and Sanitizing", san
	toxic = JoinAndSanitize(toxic_cmt, toxic_annot, san)

if aggp == 1:
	print "Reading Data Files"
	aggr_cmt = pd.read_table('convai/aggression_annotated_comments.tsv')
	aggr_annot = pd.read_table('convai/aggression_annotations.tsv')
	print "Joining and Sanitizing", san
	aggression = JoinAndSanitize(aggr_cmt, aggr_annot, san)

if attp == 1:
	print "Reading Data Files"
	attack_cmt = pd.read_table('convai/attack_annotated_comments.tsv')
	attack_annot = pd.read_table('convai/attack_annotations.tsv')
	print "Joining and Sanitizing", san
	attack = JoinAndSanitize(attack_cmt, attack_annot, san)
	

if save == 1:
	if toxp == 1:
		Tfidfize(toxic, name = 'toxic')

	if attp == 1:
		Tfidfize(attack , name = 'attack')
		
	if aggp == 1:
		Tfidfize(aggression, name = 'aggression')
		
if load == 1:
	print "Loading Models"
	if toxp == 1:
		X_toxic = sparse.load_npz("data/convai/X_toxic.npz")
		X_toxic_char = sparse.load_npz("data/convai/X_toxic_char.npz")
		X_toxic_char4 = sparse.load_npz("data/convai/X_toxic_char_cln4.npz")
		print "hStacking"
		X_toxic = sparse.hstack([X_toxic, X_toxic_char, X_toxic_char4]) 
		y_toxic = toxic['toxicity'].values

	if attp == 1:	
		X_attack = sparse.load_npz("data/convai/X_attack.npz")
		X_attack_char = sparse.load_npz("data/convai/X_attack_char.npz")
		X_attack_char4 = sparse.load_npz("data/convai/X_attack_char_cln4.npz")
		print "hStacking"
		X_attack = sparse.hstack([X_attack, X_attack_char, X_attack_char4]) 
		y_attack = attack['attack'].values

	if aggp == 1:
		X_aggression = sparse.load_npz("data/convai/X_aggression.npz")
		X_aggression_char = sparse.load_npz("data/convai/X_aggression_char.npz")
		X_aggression_char4 = sparse.load_npz("data/convai/X_aggression_char_cln4.npz")
		print "hStacking"
		X_aggression = sparse.hstack([X_aggression, X_aggression_char, X_aggression_char4]) 
		y_aggression = aggression['aggression'].values

	
	print "Fitting Models"
	ridge = Ridge()
	if toxp == 1:
		print "toxic"
		model_toxic_comb = ridge.fit(X_toxic, y_toxic)
		with open('data/convai/model1_toxic_comb4.pk', 'wb') as fin:
			pickle.dump(model_toxic_comb, fin)

	if attp == 1:
		print "attack"
		model_attack_comb = ridge.fit(X_attack, y_attack)
		with open('data/convai/model1_attack_comb4.pk', 'wb') as fin:
			pickle.dump(model_attack_comb, fin)

	if aggp == 1:
		print "aggression"
		model_aggression_comb = ridge.fit(X_aggression, y_aggression)
		with open('data/convai/model1_aggression_comb4.pk', 'wb') as fin:
			pickle.dump(model_aggression_comb, fin)


if predicting == 1:
	print "Loading Models"
	model_toxic_comb = pickle.load(open("models/convai/model1_toxic_comb4.pk", "rb"))
	model_attack_comb = pickle.load(open("models/convai/model1_attack_comb4.pk", "rb"))
	model_aggression_comb = pickle.load(open("models/convai/model1_aggression_comb4.pk", "rb"))

	print "Loading Models 2"
	tfidfer_toxic = pickle.load(open("models/convai/tfidfer_toxic.pk", "rb"))
	tfidfer_attack = pickle.load(open("models/convai/tfidfer_attack.pk", "rb"))
	tfidfer_aggression = pickle.load(open("models/convai/tfidfer_aggression.pk", "rb"))

	print "Loading Models 3"
	tfidfer_toxic_char = pickle.load(open("models/convai/tfidfer_toxic_char.pk", "rb"))
	tfidfer_attack_char = pickle.load(open("models/convai/tfidfer_attack_char.pk", "rb"))
	tfidfer_aggression_char = pickle.load(open("models/convai/tfidfer_aggression_char.pk", "rb"))

	print "Loading Models 4"
	tfidfer_toxic_char4 = pickle.load(open("models/convai/tfidfer_toxic_char_cln4.pk", "rb"))
	tfidfer_attack_char4 = pickle.load(open("models/convai/tfidfer_attack_char_cln4.pk", "rb"))
	tfidfer_aggression_char4 = pickle.load(open("models/convai/tfidfer_aggression_char_cln4.pk", "rb"))

	train_orig = pd.read_csv('input/train.csv')
	test_orig = pd.read_csv('input/test.csv')

	train_orig.fillna(' ',inplace=True) # replace NA Values from train data
	test_orig.fillna(' ',inplace=True)

	if san:
		train_orig["comment_text"] = train_orig["comment_text"].apply(lambda x: Sanitize(x))
		test_orig["comment_text"] = test_orig["comment_text"].apply(lambda x: Sanitize(x))

	def TfidfAndPredict(tfidfer, tfcharfer, tfcharfer4, model):
		tfidf_train = tfidfer.transform(train_orig['comment_text'])
		tfidf_test = tfidfer.transform(test_orig['comment_text'])

		tfcharidf_train = tfcharfer.transform(train_orig['comment_text'])
		tfcharidf_test = tfcharfer.transform(test_orig['comment_text'])

		tfcharidf_train4 = tfcharfer4.transform(train_orig['comment_text'])
		tfcharidf_test4 = tfcharfer4.transform(test_orig['comment_text'])

		tfidf_train = sparse.hstack([tfidf_train, tfcharidf_train, tfcharidf_train4]) 
		tfidf_test = sparse.hstack([tfidf_test, tfcharidf_test, tfcharidf_test4]) 

		train_scores = model.predict(tfidf_train)
		test_scores = model.predict(tfidf_test)
		return train_scores, test_scores

	print "Modelling"
	toxic_tr_scores, toxic_t_scores = TfidfAndPredict(tfidfer_toxic, tfidfer_toxic_char, tfidfer_toxic_char4,  model_toxic_comb)
	attack_tr_scores, attack_t_scores = TfidfAndPredict(tfidfer_attack, tfidfer_attack_char, tfidfer_attack_char4, model_attack_comb)
	aggression_tr_scores, aggression_t_scores = TfidfAndPredict(tfidfer_aggression, tfidfer_aggression_char, tfidfer_aggression_char4, model_aggression_comb)

	train_orig['toxic_level'] = toxic_tr_scores
	train_orig['attack'] = attack_tr_scores
	train_orig['aggression'] = aggression_tr_scores
	test_orig['toxic_level'] = toxic_t_scores
	test_orig['attack'] = attack_t_scores
	test_orig['aggression'] = aggression_t_scores

	train_orig.to_csv('input/train_convai.csv', index=False)
	test_orig.to_csv('input/test_convai.csv', index=False)