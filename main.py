from keras.layers import LSTM, Bidirectional, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from src.estimate import estimate

# All models feature an untrainable embedding layer by default
models = {
	"1hl" : [
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	"2hl" : [
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	"3hl" : [
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	"4hl" : [
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	"conv_maxPool_1hl" : [
		Conv1D(128, 5, activation="relu"),
		MaxPooling1D(5),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	"conv_maxPool_2hl" : [
		Conv1D(128, 5, activation="relu"),
		MaxPooling1D(5),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	"biDirectional_1hl" : [
		Bidirectional(LSTM(100, activation="relu", kernel_regularizer="l2")),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	"biDirectional_2hl" : [
		Bidirectional(LSTM(100, activation="relu", kernel_regularizer="l2")),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Flatten(),
		Dense(1, activation="sigmoid"),
	],
	# "1_bi_directional" : [
	# 	Bidirectional(LSTM(100, activation="relu")),
	# 	Dense(128, activation="relu"),
	# 	Dense(1, activation="sigmoid"),
	# ],
	# "2_bi_directional" : [
	# 	Bidirectional(LSTM(100, return_sequences=True, activation="relu")),
	# 	Bidirectional(LSTM(100, activation="relu")),
	# 	Dense(128, activation="relu"),
	# 	Dense(1, activation="sigmoid"),
	# ],
}

for name, layers in models.items():
	print("\n_________________________________________________________________")
	print(f"Estimating model: {name}")
	print("=================================================================\n")
	estimate(
		name=name,
		layers=layers,
		epochs=20,
	)