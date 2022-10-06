import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

def load_dataset(subset:str, embedding_method:str, tokenizer:Tokenizer, max_tokenizer_length:int):
	"""
	Loads tokenized, prefetched, cached, shuffled, and batched `Dataset` 
	from tensor slices.

	Args:
	* `subset (str)`: subset of data to load. One of "train", "dev", "test"
	or "unlabelled".
	* `tokenizer (Tokenizer)`: Tokenizer to use on input text data.
	* `max_tokenizer_length (int)`: Max length of tokenised inputs. All
	inputs are padded or truncated to fit this length.

	Returns:
	* `dataset (Dataset)`: Prefetched, cached, and shuffled Dataset from 
	tensor slices, in batches of 16.
	"""
	if embedding_method == "glove":
		print(f"Reading data from data/input/{subset}_raw.csv ...")
		df = pd.read_csv(f"data/input/{subset}_raw.csv")
		x = df["Comment"].values
		y = df["Toxicity"].values
		tokenized_x = pad_sequences(
			tokenizer.texts_to_sequences(x), 
			maxlen=max_tokenizer_length, 
			padding="post", 
			truncating="post"
		)
		dataset = tf.data.Dataset.from_tensor_slices((tokenized_x, y))
		dataset = dataset.cache()
		dataset = dataset.shuffle(160000)
		dataset = dataset.batch(64)
		dataset = dataset.prefetch(8)
		return dataset
	
	elif embedding_method == "tfidf":
		print(f"Reading data from data/input/{subset}_tfidf.csv ...")
		df = pd.read_csv(f"data/input/{subset}_tfidf.csv")
		print("Aggregating TFIDF embeddings ...")
		df["Comment"] = df[df.columns[df.columns.get_loc("Comment"):]].to_numpy().tolist()
		df["Comment"] = df["Comment"].apply(lambda x: np.asarray(x, dtype=np.float))
		x = df["Comment"].values
		y = df["Toxicity"].values
		x = list(map(list, zip(*x)))
		x = np.transpose(x)
		dataset = tf.data.Dataset.from_tensor_slices((x, y))
		dataset = dataset.cache()
		dataset = dataset.shuffle(160000)
		dataset = dataset.batch(64)
		dataset = dataset.prefetch(8)
		return dataset
	
	elif embedding_method == "bert":
		print(f"Reading data from data/input/{subset}_embedding.csv ...")
		df = pd.read_csv(f"data/input/{subset}_embedding.csv")
		print("Aggregating sentence_BERT embeddings ...")
		df["Comment"] = df[df.columns[df.columns.get_loc("Comment"):]].to_numpy().tolist()
		df["Comment"] = df["Comment"].apply(lambda x: np.asarray(x, dtype=np.float))
		x = df["Comment"].values
		y = df["Toxicity"].values
		x = list(map(list, zip(*x)))
		x = np.transpose(x)
		dataset = tf.data.Dataset.from_tensor_slices((x, y))
		dataset = dataset.cache()
		dataset = dataset.shuffle(160000)
		dataset = dataset.batch(64)
		dataset = dataset.prefetch(8)
		return dataset

def generate_embedding(max_tokenizer_length:int):
	"""
	Generates tokenizer and embedding matrix based on pre-trained GloVe
	embeddings.

	Args:
	* `max_tokenizer_length (int)`: Max length of tokenised inputs. All
	inputs are padded or truncated to fit this length.

	Returns:
	* `tokenizer (Tokenizer)`: Tokenizer to use on input text data.
	* `embedding_matrix (np.array)`: Embedding matrix of weights to apply
	to tokenized inputs. Embedding matrix based on pre-trained GloVe 
	embeddings.
	"""
	print("Tokenising data ...")
	tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>", lower=True)
	df = pd.read_csv("data/input/train_raw.csv")
	tokenizer.fit_on_texts(df["Comment"].values)
	word_index = tokenizer.word_index

	print(f"Applying GloVe embeddings ...")
	# Read GloVe embeddings
	embeddings_index = {}
	with open("data/input/glove.6B/glove.6B.100d.txt") as f:
		for line in f:
			embedding = line.split()
			embeddings_index[embedding[0]] = np.asarray(
				embedding[1:], dtype="float32"
			)
	f.close()

	# Apply embeddings
	embedding_matrix = np.zeros((len(word_index) + 1, max_tokenizer_length))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	return tokenizer, embedding_matrix


# tokenizer, embedding = generate_embedding()
# train = load_dataset(
# 	subset="train",
# 	tokenizer=tokenizer,
# 	max_tokenizer_length=100,
# )
# validation = load_dataset(
# 	subset="dev",
# 	tokenizer=tokenizer,
# 	max_tokenizer_length=100,
# )
