import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
"Asian","Atheist","Buddhist","Christian","Hindu","Jewish","Muslim","Other religion",
"Bisexual","Heterosexual","Homosexual gay or lesbian", "Other sexual orientation",
"Black", "Latino", "Other race or ethnicity", "White"
"Female","Male", "Other gender", "Transgender"
"Intellectual or learning disability", "Other disability", "Physical disability","Psychiatric or mental illness"
def evaluate(model_name, identity_subset=None):
	df = pd.read_csv(f"data/output/val/{model_name}.csv")
	if identity_subset == "religion":
		df = df.loc[
			(df["Asian"] == 1) |
			(df["Atheist"] == 1) |
			(df["Buddhist"] == 1) |
			(df["Christian"] == 1) |	
			(df["Hindu"] == 1) |
			(df["Jewish"] == 1) |
			(df["Muslim"] == 1) |
			(df["Other religion"] == 1)
		]
	elif identity_subset == "sex_orientation":
		df = df.loc[
			(df["Bisexual"] == 1) |
			(df["Heterosexual"] == 1) |
			(df["Homosexual gay or lesbian"] == 1) |
			(df["Other sexual orientation"] == 1) 	
		]
	elif identity_subset == "race":
		df = df.loc[
			(df["Black"] == 1) |
			(df["Latino"] == 1) |
			(df["Other race or ethnicity"] == 1) |
			(df["White"] == 1) 	
		]
	elif identity_subset == "gender":
		df = df.loc[
			(df["Male"] == 1) |
			(df["Female"] == 1) |
			(df["Other gender"] == 1) |
			(df["Transgender"] == 1) 	
		]
	elif identity_subset == "disability":
		df = df.loc[
			(df["Intellectual or learning disability"] == 1) |
			(df["Other disability"] == 1) |
			(df["Physical disability"] == 1) |
			(df["Psychiatric or mental illness"] == 1) 	
		]

	print("===========================================")
	print(len(df))

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
		model_name=model,
		identity_subset="disability"
	)
	results.append({
		"Model": model,
		"Precision": precision,
		"Accuracy": accuracy,
		"Recall": recall,
		"F-score": f1,
	})

df = pd.DataFrame(results)

print(df.head(50))
print(df.to_latex(index=False))  
