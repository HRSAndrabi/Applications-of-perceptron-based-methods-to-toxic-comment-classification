import sklearn as sk
import pandas as pd
import numpy as np
# from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords

def load_df(dataset, embedding_method:str="glove", glove_aggregation_func:str="avg"):
	# APPLY GLOVE EMBEDDINGS
	if embedding_method == "glove":
		df = pd.read_csv(f"data/input/{dataset}_raw.csv")
		print("Tokenising data ...")
		df["Comment"] = df["Comment"].str.replace("-", "")
		df["Comment"] = df["Comment"].str.replace('[^\w\s]','')
		print(df["Toxicity"].value_counts())

		# Read GloVe embeddings
		print(f"Reading GloVe embeddings ...")
		embeddings_index = {}
		with open("data/input/glove.6B/glove.6B.100d.txt") as f:
			for line in f:
				embedding = line.split()
				embeddings_index[embedding[0]] = np.asarray(
					embedding[1:], dtype="float32"
				)
		f.close()

		# Apply embeddings
		print("Applying GloVe embeddings ...")
		df["Comment"] = df["Comment"].apply(apply_glove, args=(embeddings_index,glove_aggregation_func,))
		return df, df["Comment"].to_list(), df["Toxicity"].to_list()
	
	elif embedding_method == "tfidf":
		print(f"Reading {dataset} data ...")
		df = pd.read_csv(f"data/input/{dataset}_tfidf.csv")
		print("Aggregating TFIDF embeddings ...")
		embedding = df[df.columns[df.columns.get_loc("Comment"):]].to_numpy().tolist()
		df["Comment"] = df[df.columns[df.columns.get_loc("Comment"):]].to_numpy().tolist()
		return df, df["Comment"].to_list(), df["Toxicity"].to_list()

	elif embedding_method == "bert":
		print(f"Reading {dataset} data ...")
		df = pd.read_csv(f"data/input/{dataset}_embedding.csv")
		print("Aggregating BERT embeddings ...")
		embedding = df[df.columns[df.columns.get_loc("Comment"):]].to_numpy().tolist()
		df["Comment"] = df[df.columns[df.columns.get_loc("Comment"):]].to_numpy().tolist()
		return df, df["Comment"].to_list(), df["Toxicity"].to_list()

def apply_glove(comment, embeddings_index, glove_aggregation_func):
	comment = [
		str.lower(word) for word in comment.split() 
		if str.lower(word) not in (stopwords.words("english")) and word.isalpha()
	]
	if len(comment) <= 100:
		comment += ["<OOV>"] * (100 - len(comment))
	else:
		comment = comment[:100]
	vector_representation = []
	for word in comment:
		vector = embeddings_index.get(word)
		if vector is not None:
			vector_representation.append(vector)
		else:
			vector_representation.append(
				np.zeros(len(embeddings_index["the"]))
			)
	vector_representation = np.array(vector_representation)

	if glove_aggregation_func == "min":
		return np.amin(vector_representation, axis=0)
	elif glove_aggregation_func == "max":
		return np.amax(vector_representation, axis=0)
	else:
		return np.average(vector_representation, axis=0)

def estimate(embedding_method:str="glove", ouput_file_name:str="",  glove_aggregation_func:str="avg"):
	train_df, x_train, y_train = load_df(
		dataset="train",
		embedding_method=embedding_method,
		glove_aggregation_func=glove_aggregation_func,
	)
	val_df, x_val, y_val = load_df(
		dataset="dev",
		embedding_method=embedding_method,
		glove_aggregation_func=glove_aggregation_func,
	)
	test_df, x_test, y_test = load_df(
		dataset="test",
		embedding_method=embedding_method,
		glove_aggregation_func=glove_aggregation_func,
	)

	model = LogisticRegression()
	model.fit(x_train, y_train)

	print(f"Train accuracy: {model.score(x_train, y_train)}")
	print(f"Validation accuracy: {model.score(x_val, y_val)}")

	val_df["Toxicity_pred"] = model.predict(x_val)
	val_df.drop(
		["Comment"], axis=1
	).to_csv(
		path_or_buf=f"data/output/val/{ouput_file_name}.csv", 
		index=False
	)

	val_df["Toxicity_pred"] = model.predict(x_test)
	test_df.drop(
		["Comment"], axis=1
	).to_csv(
		path_or_buf=f"data/output/test/{ouput_file_name}.csv", 
		index=False
	)

estimate(
	embedding_method="bert",
	ouput_file_name="logistic_bert"
)