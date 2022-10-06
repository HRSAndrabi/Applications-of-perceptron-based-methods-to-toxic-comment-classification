from keras.models import Sequential
from keras.layers import Embedding
from keras.metrics import Precision, Recall, AUC
from keras.initializers import Constant

def construct_model(embedding, embedding_method, max_tokenizer_length:int, layers:list):
	"""
	Constructs model based on supplied embeddings weights and intermediate layers.
	All models feature an embedding layer by default.

	Args:
	* `embedding (np.array)`: Untrainable embedding weights to supply to the 
	embedding layer.
	* `max_tokenizer_length (int)`: Max length of tokenised inputs. All
	inputs are padded or truncated to fit this length.
	* `layers (list)`: An array of layers to add to the model. All models are 
	initialised with an untrainable embedding layer by default.

	Returns:
	* `model (Sequential)`: Sequential model of specified architecture.
	"""
	model = Sequential()
	if embedding_method == "glove":
		model.add(Embedding(
			input_dim=len(embedding),
			output_dim=max_tokenizer_length,
			# weights=[embedding],
			embeddings_initializer=Constant(embedding),
			input_length=max_tokenizer_length,
			trainable=False
		))
	for layer in layers:
		model.add(layer)

	model.compile(loss="BinaryCrossentropy", optimizer="Adam", metrics=["acc", Precision(), Recall(), AUC()])
	model.summary()
	return model
