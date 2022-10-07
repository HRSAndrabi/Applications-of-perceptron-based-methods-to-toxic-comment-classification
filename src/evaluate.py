import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def evaluate(model_name):
	df = pd.read_csv(f"data/output/val/{model_name}.csv")
	df.head(12)

	y_true = df["Toxicity"]
	y_pred = df["Toxicity_pred"]
	precision = precision_score(y_true, y_pred, average="binary")
	recall = recall_score(y_true, y_pred, average="binary")
	accuracy = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average="binary")

	return round(precision,4), round(recall,4), round(accuracy,4), round(f1, 4)

models = [
	"logistic_tfidf", "logistic_bert", "logistic_glove_min", "logistic_glove_max", "logistic_glove_avg", 
	"1hl_tfidf", "1hl_bert", "1hl_glove",
	"2hl_tfidf", "2hl_bert", "2hl_glove",
	"conv_maxPool_1hl_tfidf", "conv_maxPool_1hl_bert", "conv_maxPool_1hl_glove", 
]
results = []
for model in models:
	print(model)
	precision, recall, accuracy, f1 = evaluate(
		model_name=model
	)
	results.append({
		"Model": model,
		"Precision": precision,
		"Accuracy": accuracy,
		"Recall": recall,
		"F-score": f1,
	})

df = pd.DataFrame(results)

print(results)
print(df.head(50))
print(df.style.to_latex(index=False))  