import pandas as pd 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

def plot_word_frequency(dataset, num_words:int=50):
	df = pd.read_csv(f"data/input/{dataset}")
	df["Comment"] = df["Comment"].apply(
		lambda x: ' '.join([str.lower(word) for word in x.split() if str.lower(word) not in (stopwords.words("english"))])
	)
	df["Comment"] = df["Comment"].str.replace("-", "")
	df["Comment"] = df["Comment"].str.replace('[^\w\s]','')

	toxic_df = df.copy()
	toxic_df = toxic_df.loc[
		toxic_df["Toxicity"] == 1
	]
	non_toxic_df = df.copy()
	non_toxic_df = non_toxic_df.loc[
		non_toxic_df["Toxicity"] == 0
	]

	toxic_words = toxic_df["Comment"].str.split(expand=True).stack().value_counts()
	non_toxic_words = non_toxic_df["Comment"].str.split(expand=True).stack().value_counts()

	with plt.style.context('science'):
		for plot in ["toxic", "non_toxic"]:
			if plot == "toxic":
				file_name = "toxic_word_freq.pdf"
				bars = toxic_words
			else:
				file_name = "nontoxic_word_freq.pdf"
				bars = non_toxic_words

			plt.title(f"Most frequent words in {plot} comments")
			plt.figure(figsize=(9,3))
			plt.ylabel("Frequency")
			# plt.xlabel("Words")
			plt.bar(bars.index.values[:num_words], bars.values[:num_words])
			plt.gca().tick_params(axis="x", which="both", length=0)
			plt.gca().tick_params(axis="y", which="minor", length=0)
			plt.xticks(rotation = 90)
			plt.savefig(
				fname=f"./manuscript/graphics/{file_name}",
				dpi=300,
			)

plot_word_frequency(
	dataset="dev_raw.csv",
	num_words=25
)