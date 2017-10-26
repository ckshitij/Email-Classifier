# Email Classifier 
- Email Classifier is a model which classify that which type of your mail it is, for example like whether mail is *spam or not*. 
- Its gives the label to the mail like in gmail we have many category like update ,spam, normal mails etc.

___________________
## Module Used Documentation
- [Scikit Learn](http://scikit-learn.org/stable/documentation.html)
- [Nltk](http://www.nltk.org/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/)
- [Numpy](https://docs.scipy.org/doc/numpy/reference/)
- [Gensim](https://radimrehurek.com/gensim/tutorial.html)
- [Fasttext](https://github.com/facebookresearch/fastText)
- [Fasttext Python Wrapper](https://pypi.python.org/pypi/fasttext)
- [Re](https://docs.python.org/2/library/re.html)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [LIME](https://github.com/marcotcr/lime)
- [XG Boost](http://xgboost.readthedocs.io/en/latest/model.html)
## Problem On Installing Libraries in Windows
- Open __CMD__ 
	- _pip_ install -lib name
- If _pip_ not Worked
	- _Download_ the Repository From Github.
	- _Extarct_ the module
	- _Copy_ module and _Paste_ it into python Folder 
		- Then Open __CMD__
		```sh
		cd package_name
		python setup.py install
		```
	- If it required Dependency then do the Same for other dependencies modules.
- Before Installing Tensorflow or any _Deep learning_ Module make sure you have Installed __Glib module and C++ (Latest Version)__ .
- On Installing __Gensim__ we Required __Pattern__ Library for _Lemmatization_ which is in python2.
- On Using Keras by default it uses tensorflow as a backend.
	- There are three option 
		- tensorflow
		- theano
		- cltk
	- For changing backend of keras :-
		- [Changing Backend Of Keras](https://keras.io/backend/)

## Working Model 

- ### Data Normalization
	- Merge All the files by using Pandas DataFrames  
    	```python
	    df1 = pd.read_excel(file1,index_col=None, na_values=['NA'], parse_cols = [])
	    df2 = pd.read_excel(file2,index_col=None, na_values=['NA'], parse_cols = [])
	    frames = [df1,df2]
        df = pd.concat(frames,ignore_index=True)
        df.to_csv(mainfile,encoding='utf-8',index = False)
	    ```
	- Preprocess Mails
		- Remove Html Components by [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
		- Remove Disclaimer if Any 
		- Remove some words , punctuations etc using Regex by [re](https://docs.python.org/2/library/re.html)
	- Object Standardization
		```python
		stan_dic = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love", "..."}
		def lookup_words(Mail):
			words = Mail.split()
			word_n = []
			for word in words:
				if word in stan_dic:
					word = stan_dic[word]
				word_n.append(word)
			an = " ".join(str(x) for x in word_n)
			an = an.lower()
			del words[:]
			del word_n[:]
			return an

		lookup_words("RT this is a retweeted tweet by Shivam Bansal")
		>> "Retweet this is a retweeted tweet by Shivam Bansal"
		```
	- __Efficient stopword removal__ : POS tags are also useful in efficient removal of stopwords.
	- __Normalization and Lemmatization__: POS tags are the basis of lemmatization process for converting a word to its base form (lemma).
		```python
		def tokenize(document): 
			document = unicode(document,'utf-8')
			lemmy = []
			for sent in sent_tokenize(document):
				for token, tag in pos_tag(wordpunct_tokenize(sent)):
					if token in stoplist:
						continue
					lemma = lemmatize(token, tag)
					lemmy.append(lemma)
			return lemmy

		def lemmatize(token, tag):
			tag = {
				'N': wn.NOUN,
				'V': wn.VERB,
				'R': wn.ADV,
				'J': wn.ADJ
			}.get(tag[0], wn.NOUN)
			lemmatizer = WordNetLemmatizer()
			return lemmatizer.lemmatize(token, tag)
		```
	- Save the Processed file  
		- df.to_csv(filename,encoding='utf-8',index = False)
		- _(index = Flase)_ Avoid to Create new Index again.
- ### Statistical Features
	-  __Term Frequency – Inverse Document Frequency (TF – IDF)__
		```python
		vectorizer = TfidfVectorizer(max_df=0.5, max_features=15000, min_df=2, stop_words='english', use_idf=True , ngram_range=(1,3))
		```
		- __N-Grams__ :-  A combination of N words together are called N-Grams.<br>
						  N grams (N > 1) are generally more informative as compared to words (Unigrams) as features.<br>
						  Also, bigrams (N = 2) are considered as the most important features of all the others.<br>
		- __max_df__  :-  float in range [0.0, 1.0] or int, default=1.0 <br>
						  When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).<br> 
						  If float, the parameter represents a proportion of documents, integer absolute counts.<br>
						  If you set max_df = 0.6 then that would translate to 0.6*5=3 documents.<br> 
						  If you set max_df = 2 then that would simply translate to 2 documents.<br>
	    - __min_df__  :-  float in range [0.0, 1.0] or int, default=1.0 <br>
						  When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.<br> 
						  This value is also called cut-off in the literature.<br>
						  If float, the parameter represents a proportion of documents, integer absolute counts.<br>
						  If you set min_df = 0.6 then that would translate to 0.6*5=3 documents.<br> 
						  If you set min_df = 2 then that would simply translate to 2 documents.<br>  
	-  __Latent Semantic Analysis (LSA)__ :- It is a technique in natural language processing, in particular distributional semantics, of analyzing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms.<br> 
											  LSA assumes that words that are close in meaning will occur in similar pieces of text __(the distributional hypothesis)__.<br> 
											  A matrix containing word counts per paragraph (rows represent unique words and columns represent each paragraph) is constructed from a large piece of text and a mathematical technique called __singular value decomposition (SVD)__ is used to reduce the number of rows while preserving the similarity structure among columns. <br>
											  Words are then compared by taking the cosine of the angle between the two vectors (or the dot product between the normalizations of the two vectors) formed by any two rows.<br>
											  Values close to 1 represent very similar words while values close to 0 represent very dissimilar words.
		
		```python
		svd = TruncatedSVD(200,n_iter=7, random_state=42)
		lsa = make_pipeline(vectorizer,svd, Normalizer(copy=False))
		```
		- [Singular Value Decomposition (SVD)](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- ### Supervised Model
	- #### Extra tree Classifier
		```python
		clf = ExtraTreesClassifier(n_estimators=200,n_jobs=-1,max_depth=36,class_weight=data_dic)
		pipe = make_pipeline(lsa, clf)
		pipe.fit(training_data,target)
		```
		- [About Extra tree Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
		- [About PipeLine](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
	- __Saving Model__
		- [Using Joblib lib](https://pypi.python.org/pypi/joblib)
			```python
			from sklearn.externals import joblib
			filename = 'final_model.pkl'
			joblib.dump(clf,filename)
			
			filename = 'tfidf_model.pkl'
			joblib.dump(vect,filename)
			
			filename = 'lsa_model.pkl'
			joblib.dump(lsa,filename)
			```
- ### Local Interpretable Model –Agnostic Explanation (LIME)
	- Lime is able to explain any black box text classifier, with two or more classes. 
	- All we require is that the classifier implements a function that takes in raw text or a numpy array and outputs a probability for each class. 
	- Support for scikit-learn classifiers is built-in.
		```python
		from lime.lime_text import LimeTextExplainer
		explainer = LimeTextExplainer(class_names = class_name)
		exp = explainer.explain_instance(doc,pipe.predict_proba,num_features=6,top_labels=4)
		exp.show_in_notebook(text = False)
		```
## Implementation Of XgBoost
- [About Xg Boost](http://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)
	```python
	from xgboost.sklearn import XGBClassifier
	model = XGBClassifier(objective='multi:softmax',nthread=4,n_estimators=1000,scale_pos_weight=data_dic)
	model.fit(Lsa_training_data , target)
	# make predictions for test data
	y_pred = model.predict(Lsa_test_data)
	```
	
## Implementation Of DNNClassifier
- [About Dnn Classifier](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier)
	```python
	import tensorflow.contrib.learn as ln
	feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(Lsa_training_data )
	clf = ln.DNNClassifier(hidden_units=[],optimizer = tf.train.RMSPropOptimizer(),dropout=0.2,n_classes=,feature_columns=feature_columns,model_dir="$PATH")
	clf.fit(Lsa_training_data ,y_train,batch_size=256,max_steps=40000)
	```
	- [RMSPropOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer)
	- [Dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks))
	- $PATH (Where you have to store your model)
	- Refer to Refrence no. 5
	
## References
1. [Pattern3](https://github.com/pattern3/pattern)
2. [NLTK Word Processing](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
3. [LSA Implemntation by Chris jmc cormick](https://github.com/chrisjmccormick/LSA_Classification)
4. [CNNs for sentence classification](https://github.com/yoonkim/CNN_sentence)
5. [CNN, RNN (GRU and LSTM) and Word Embeddings on Tensorflow](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn)
6. [HackerEarth Data Cleaning Tutorial](https://www.hackerearth.com/practice/machine-learning/advanced-techniques/text-mining-feature-engineering-r/tutorial/)
