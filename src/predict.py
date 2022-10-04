from keras.utils import pad_sequences
import pandas as pd

def make_predictions(model_name:str, model, tokenizer):
	"""
	Generates prediction labels for test_raw.csv using input `Model`.

	Args:
	* model_name (str): Name of the model.
	* `model (Sequential)`: A trained keras.models.Sequential model.
	* `tokenizer (Tokenizer)`: Tokenizer to use on input text data.
	"""
	print("Predicting test dataset ...")
	test_dataset = pd.read_csv("data/input/test_raw.csv")
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
	test_dataset["Toxicity"] = y_pred
	test_dataset.to_csv(f"data/output/{model_name}.csv", columns=["ID", "Toxicity"], index=False)

# model = tf.keras.models.load_model("data/models/2conv_2maxPool_2hl.h5")