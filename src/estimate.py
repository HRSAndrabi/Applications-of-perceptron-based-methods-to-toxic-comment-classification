import src.construct_model as construct_model
import src.preprocess as preprocess
import src.predict as predict
from keras.callbacks import CSVLogger
from keras.metrics import Precision, Recall, Accuracy
import os
import numpy as np


def estimate(name:str, layers:list, tokenizer, embedding, embedding_method, max_tokenizer_length:int=100, epochs:int=1):
	"""
	A method to estimate a model with provided layer specification. Trained model
	outputs and logs are saved under data/output/model

	Args:
	* `name (str)`: Name of the model. Output files will be saved under this name.
	* `layers (list)`: An array of layers to add to the model. All models are 
	initialised with an untrainable embedding layer by default.
	* `tokenizer (Tokenizer)`: Tokenizer to use on input text data.
	* `embedding (np.array)`: Untrainable embedding weights to supply to the 
	embedding layer.
	* `max_tokenizer_length (int, optional)`: Max length of tokenised inputs. All
	inputs are padded or truncated to fit this length. Defaults to 100.
	* `epochs (int, optional)`: Number of epochs for which to train the model. 
	Defaults to 1.

	Returns:
	* `model (Sequential)`: A trained keras.models.Sequential model.
	* `history`: Training history of the model.
	"""
	train = preprocess.load_dataset(
		subset="train",
		embedding_method=embedding_method,
		tokenizer=tokenizer,
		max_tokenizer_length=max_tokenizer_length,
	)
	validation = preprocess.load_dataset(
		subset="dev",
		tokenizer=tokenizer,
		embedding_method=embedding_method,
		max_tokenizer_length=max_tokenizer_length,
	)
	model = construct_model.construct_model(
		embedding=embedding,
		embedding_method=embedding_method,
		max_tokenizer_length=max_tokenizer_length,
		layers=layers,
	)

	logger_path = f"./data/models/history/{name}.log"
	if not os.path.exists(logger_path):
		open(logger_path, 'w').close()
	logger = CSVLogger(logger_path, separator=",", append=False)

	history = model.fit(train, epochs=epochs, validation_data=validation, callbacks=[logger])
	model.save(f"./data/models/{name}.h5")

	predict.make_predictions(
		model_name=name,
		model=model,
		subset="dev",
		tokenizer=tokenizer,
		embedding_method=embedding_method
	)

	pre = Precision()
	re = Recall()
	acc = Accuracy()
	for batch in validation.as_numpy_iterator(): 
		x, y_true = batch
		y_pred = (model.predict(x) > 0.5).astype(int)
		
		x = x.flatten()
		y_true = y_true.flatten()
		y_pred = y_pred.flatten()
		
		pre.update_state(y_true, y_pred)
		re.update_state(y_true, y_pred)
		acc.update_state(y_true, y_pred)

	print(f"Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}")

	return model, history
