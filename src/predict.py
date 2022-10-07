from keras.utils import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

def make_predictions(model_name:str, subset, model, embedding_method, tokenizer=None):
	"""
	Generates prediction labels for test_raw.csv using input `Model`.

	Args:
	* `model_name (str)`: Name of the model.
	* `model (Sequential)`: A trained keras.models.Sequential model.
	* `tokenizer (Tokenizer)`: Tokenizer to use on input text data.
	"""
	print("Predicting test dataset ...")
	if embedding_method == "glove":
		test_dataset = pd.read_csv(f"data/input/{subset}_raw.csv")
		x = test_dataset["Comment"].values
		tokenized_x = pad_sequences(
			tokenizer.texts_to_sequences(x), 
			maxlen=100, 
			padding="post", 
			truncating="post"
		)
		y_pred = (model.predict(tokenized_x) > 0.5).astype(int)
		x = x.flatten()
		y_pred = y_pred.flatten()
		test_dataset["Toxicity_pred"] = y_pred
		test_dataset = test_dataset.loc[:, ~test_dataset.columns.str.contains('^Unnamed')]
		# test_dataset.to_csv(f"data/output/{subset}/{model_name}.csv", columns=["ID", "Toxicity"], index=False)
		test_dataset.drop(["Comment"], axis=1).to_csv(f"data/output/val/{model_name}.csv", index=False)
	
	elif embedding_method == "tfidf":
		test_dataset = pd.read_csv(f"data/input/{subset}_tfidf.csv")
		print("Aggregating TFIDF embeddings ...")
		test_dataset["Comment"] = test_dataset[test_dataset.columns[test_dataset.columns.get_loc("Comment"):]].to_numpy().tolist()
		x = test_dataset["Comment"].values
		test_dataset["Comment"] = test_dataset["Comment"].apply(lambda x: np.asarray(x, dtype=np.float))
		x = test_dataset["Comment"].values
		x = list(map(list, zip(*x)))
		x = np.transpose(x)
		print(len(x))
		print(len(x[0]))
		y_pred = (model.predict(x) > 0.5).astype(int)
		test_dataset["Toxicity_pred"] = y_pred
		test_dataset = test_dataset.loc[:, ~test_dataset.columns.str.contains('^Unnamed')]
		test_dataset.drop(["Comment"], axis=1).to_csv(f"data/output/val/{model_name}.csv", index=False)

	elif embedding_method == "bert":
		test_dataset = pd.read_csv(f"data/input/{subset}_embedding.csv")
		print("Aggregating BERT embeddings ...")
		test_dataset["Comment"] = test_dataset[test_dataset.columns[test_dataset.columns.get_loc("Comment"):]].to_numpy().tolist()
		x = test_dataset["Comment"].values
		test_dataset["Comment"] = test_dataset["Comment"].apply(lambda x: np.asarray(x, dtype=np.float))
		x = test_dataset["Comment"].values
		x = list(map(list, zip(*x)))
		x = np.transpose(x)
		print(len(x))
		print(len(x[0]))
		y_pred = (model.predict(x) > 0.5).astype(int)
		test_dataset["Toxicity_pred"] = y_pred
		test_dataset = test_dataset.loc[:, ~test_dataset.columns.str.contains('^Unnamed')]
		test_dataset.drop(["Comment"], axis=1).to_csv(f"data/output/val/{model_name}.csv", index=False)
		


# model = tf.keras.models.load_model("data/models/1hl_bert.h5")
# make_predictions(
# 	model_name="1hl_bert",
# 	model=model,
# 	subset="dev",
# 	tokenizer=None,
# 	embedding_method="bert"
# )