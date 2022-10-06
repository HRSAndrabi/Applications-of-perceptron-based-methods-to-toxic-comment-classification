from keras.layers import LSTM, Bidirectional, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten, InputLayer
from src.estimate import estimate
import src.preprocess as preprocess

# All models feature an untrainable embedding layer by default
models = {
	"1hl_bert" : [
		InputLayer(input_shape=(1000,)),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	"2hl_bert" : [
		InputLayer(input_shape=(1000,)),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	# "3hl" : [
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "4hl" : [
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	"conv_maxPool_1hl_bert" : [
		InputLayer(input_shape=(1000,)),
		Conv1D(128, 5, activation="relu"),
		MaxPooling1D(5),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	# "conv_maxPool_2hl" : [
	# 	Conv1D(128, 5, activation="relu"),
	# 	MaxPooling1D(5),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "2conv_2maxPool_1hl" : [
	# 	Conv1D(128, 5, activation="relu"),
	# 	MaxPooling1D(5),
	# 	Conv1D(128, 5, activation="relu"),
	# 	MaxPooling1D(5),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "2conv_2maxPool_2hl" : [
	# 	Conv1D(128, 5, activation="relu"),
	# 	MaxPooling1D(5),
	# 	Conv1D(128, 5, activation="relu"),
	# 	MaxPooling1D(5),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "3conv_1maxPool_1hl" : [
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	MaxPooling1D(5),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "3conv_1maxPool_2hl" : [
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	MaxPooling1D(5),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "4conv_1maxPool_1hl" : [
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	MaxPooling1D(5),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "4conv_1maxPool_2hl" : [
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	Conv1D(128, 5, activation="relu"),
	# 	MaxPooling1D(5),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "lstm_1hl" : [
	# 	Bidirectional(LSTM(100, activation="relu", kernel_regularizer="l2")),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "lstm_2hl" : [
	# 	Bidirectional(LSTM(100, activation="relu", kernel_regularizer="l2")),
	# 	Dense(128, activation="relu", kernel_regularizer="l2"),
	# 	Flatten(),
	# 	Dense(1, activation="sigmoid"),
	# ],
}

for name, layers in models.items():
	max_tokenizer_length = 100
	tokenizer, embedding = preprocess.generate_embedding(
		max_tokenizer_length=max_tokenizer_length,
	)
	print("\n_________________________________________________________________")
	print(f"Estimating model: {name}")
	print("=================================================================\n")
	try:
		estimate(
			name=name,
			layers=layers,
			tokenizer=tokenizer,
			embedding=embedding,
			epochs=10,
			max_tokenizer_length=max_tokenizer_length,
			embedding_method="tfidf",
		)
	except Exception as e:
		print(e)
