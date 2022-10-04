import src.construct_model as construct_model
import src.preprocess as preprocess
from keras.callbacks import CSVLogger
from keras.metrics import Precision, Recall, CategoricalAccuracy
import os


def estimate(name:str, layers:list, max_tokenizer_length:int=100, epochs:int=1):
	"""
	A method to estimate a model with provided layer specification. Trained model
	outputs and logs are saved under data/output/model

	Args:
	* `name (str)`: Name of the model. Output files will be saved under this name.
	* `layers (list)`: An array of layers to add to the model. All models are 
	initialised with an untrainable embedding layer by default.
	* `max_tokenizer_length (int, optional)`: Max length of tokenised inputs. All
	inputs are padded or truncated to fit this length. Defaults to 100.
	* `epochs (int, optional)`: Number of epochs for which to train the model. 
	Defaults to 1.

	Returns:
	* `model (Sequential)`: A trained keras.models.Sequential model.
	* `history`: Training history of the model.
	"""
	tokenizer, embedding = preprocess.generate_embedding(
		max_tokenizer_length=max_tokenizer_length,
	)
	train = preprocess.load_dataset(
		subset="train",
		tokenizer=tokenizer,
		max_tokenizer_length=max_tokenizer_length,
	)
	validation = preprocess.load_dataset(
		subset="dev",
		tokenizer=tokenizer,
		max_tokenizer_length=max_tokenizer_length,
	)
	model = construct_model.construct_model(
		embedding=embedding,
		max_tokenizer_length=max_tokenizer_length,
		layers=layers,
	)

	logger_path = f"./data/models/history/{name}.log"
	if not os.path.exists(logger_path):
		open(logger_path, 'w').close()
	logger = CSVLogger(logger_path, separator=",", append=False)

	history = model.fit(train, epochs=epochs, validation_data=validation, callbacks=[logger])
	model.save(f"./data/models/{name}.h5")

	pre = Precision()
	re = Recall()
	acc = CategoricalAccuracy()

	for batch in validation.as_numpy_iterator(): 
		X_true, y_true = batch
		yhat = model.predict(X_true)
		y_true = y_true.flatten()
		yhat = yhat.flatten()
		
		pre.update_state(y_true, yhat)
		re.update_state(y_true, yhat)
		acc.update_state(y_true, yhat)

	print(f"Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}")

	return model, history
