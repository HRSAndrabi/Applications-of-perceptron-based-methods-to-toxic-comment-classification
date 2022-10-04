from keras.layers import LSTM, Bidirectional, Dense
from src.estimate import estimate

# All models feature an untrainable embedding layer by default
models = {
	"1_bi_directional_regularised" : [
		Bidirectional(LSTM(100, activation="relu")),
		Dense(128, activation="relu", kernel_regularizer="l2"),
		Dense(1, activation="sigmoid"),
	],
	"1_bi_directional" : [
		Bidirectional(LSTM(100, activation="relu")),
		Dense(128, activation="relu"),
		Dense(1, activation="sigmoid"),
	],
	"2_bi_directional" : [
		Bidirectional(LSTM(100, return_sequences=True, activation="relu")),
		Bidirectional(LSTM(100, activation="relu")),
		Dense(128, activation="relu"),
		Dense(1, activation="sigmoid"),
	],
}

for name, layers in models.items():
	estimate(
		name=name,
		layers=layers,
		epochs=10,
	)