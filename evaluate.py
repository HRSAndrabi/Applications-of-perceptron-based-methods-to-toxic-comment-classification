import tensorflow as tf
import numpy as np
import src.preprocess as preprocess
from keras.metrics import Precision, Recall, Accuracy

model = tf.keras.models.load_model("data/models/simple.h5")
print(model.summary())

tokenizer, embedding = preprocess.generate_embedding(
	max_tokenizer_length=100
)
validation = preprocess.load_dataset(
	subset="dev",
	tokenizer=tokenizer,
	max_tokenizer_length=100,
)

pre = Precision()
re = Recall()
acc = Accuracy()

for batch in validation.as_numpy_iterator(): 
	x, y_true = batch
	y_pred = model.predict(x)
	y_pred = np.where(y_pred >= 0.5, 1, 0)
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	# print(y_true)
	# print(y_pred)
	print(zip(y_true, y_pred))
	
	pre.update_state(y_true, y_pred)
	re.update_state(y_true, y_pred)
	acc.update_state(y_true, y_pred)

print(f"Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}")