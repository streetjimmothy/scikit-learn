import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF

import time
import humanize
import psutil

dataset = fetch_20newsgroups(
	remove=("headers", "footers", "quotes"),
	subset="all",
	shuffle=True,
	random_state=42,
)
labels = dataset.target
unique_labels, category_sizes = np.unique(labels, return_counts=True)

true_k = unique_labels.shape[0]
print(f"{len(dataset.data)} documents - {true_k} categories")

evaluations = []
evaluations_std = []

def fit_and_evaluate(km, X, name=None, n_runs=5):
	name = km.__class__.__name__ if name is None else name

	train_times = []
	scores = [] #score = "Silhouette Coefficient"
	for seed in range(n_runs):
		km.set_params(random_state=seed)
		start_time = time.time()
		km.fit(X)
		train_times.append(time.time() - start_time)
		scores.append(
			metrics.silhouette_score(X, km.labels_, sample_size=4000)
		)
	train_times = np.asarray(train_times)

	print(f"clustering done in {humanize.precisedelta(train_times.mean(), suppress=['days', 'microseconds'])} ± {humanize.precisedelta(train_times.std(), suppress=['days', 'microseconds'])} ")
	evaluation = {
		"estimator": name,
		"train_time": train_times.mean(),
	}
	evaluation_std = {
		"estimator": name,
		"train_time": train_times.std(),
	}
	mean_score, std_score = np.mean(scores), np.std(scores)
	print(f"Silhouette Coefficient: {mean_score:.3f} ± {std_score:.3f}")
	evaluations.append(evaluation)
	evaluations_std.append(evaluation_std)

def plot_top_words(model, feature_names, n_top_words, title):
	fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
	axes = axes.flatten()
	for topic_idx, topic in enumerate(model.components_):
		top_features_ind = topic.argsort()[-n_top_words:]
		top_features = feature_names[top_features_ind]
		weights = topic[top_features_ind]

		ax = axes[topic_idx]
		ax.barh(top_features, weights, height=0.7)
		ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
		ax.tick_params(axis="both", which="major", labelsize=20)
		for i in "top right left".split():
			ax.spines[i].set_visible(False)
		fig.suptitle(title, fontsize=40)

	plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
	plt.show()

TFIDFvectorizer = TfidfVectorizer(
	min_df=0.05,
	stop_words="english",
)
start_time = time.time()
TFIDFvectorised_dataset = TFIDFvectorizer.fit_transform(dataset.data)

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
TFvectorizer = CountVectorizer(
	max_df=0.95, min_df=2, stop_words="english"
)
TFvectorised_dataset = TFvectorizer.fit_transform(dataset.data)
print(f"vectorization done in {humanize.precisedelta(time.time() - start_time, suppress=['days', 'microseconds'])}")
print(f"n_samples/documents: {TFIDFvectorised_dataset.shape[0]}, n_features/words: {TFIDFvectorised_dataset.shape[1]}") #shape is rows of the matrix in that dimension
print(f"Sparsity (number of cells with non-zero values): {TFIDFvectorised_dataset.nnz / np.prod(TFIDFvectorised_dataset.shape):.3f}")

print("Kmeasn clustering...")
start_time = time.time()
for seed in range(5):
	kmeans = KMeans(
		n_clusters=true_k,
		max_iter=100,
		n_init=1,
		random_state=seed,
	).fit(TFIDFvectorised_dataset)
	cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
	print(f"Number of elements assigned to each cluster: {cluster_sizes}")
print()
print(
	"True number of documents in each category according to the class labels: "
	f"{category_sizes}"
)

kmeans = KMeans(
	n_clusters=true_k,
	max_iter=100,
	n_init=10,
)

fit_and_evaluate(kmeans, TFIDFvectorised_dataset, name="KMeans on tf-idf vectors")
print(f"kmeans clustering done in {humanize.precisedelta(time.time() - start_time, suppress=['days', 'microseconds'])}")

# Fit the NMF models

n_components = 20
init = "nndsvda"
batch_size = 256

print("Fitting the NMF model (Frobenius norm) with tf-idf features")
start_time = time.time()
FrobNMF = NMF(
	n_components=n_components,
	random_state=1,
	init=init,
	beta_loss="frobenius",
	solver="mu",
	alpha_W=0.00005,
	alpha_H=0.00005,
	l1_ratio=1,
).fit(TFIDFvectorised_dataset)
print(f"Done in {humanize.precisedelta(time.time() - start_time, suppress=['days', 'microseconds'])}")

print("\n", "Fitting the NMF model (generalized Kullback-Leibler divergence) with tf-idf features")
KLNMF = NMF(
	n_components=20,
	random_state=1,
	init=init,
	beta_loss="kullback-leibler",
	solver="mu",
	max_iter=1000,
	alpha_W=0.00005,
	alpha_H=0.00005,
	l1_ratio=0.5,
).fit(TFIDFvectorised_dataset)
print(f"Done in {humanize.precisedelta(time.time() - start_time, suppress=['days', 'microseconds'])}")

print("\n", "Fitting the MiniBatchNMF model (Frobenius norm) with tf-idf")
FrobMBNMF = MiniBatchNMF(
	n_components=n_components,
	random_state=1,
	batch_size=batch_size,
	init=init,
	beta_loss="frobenius",
	alpha_W=0.00005,
	alpha_H=0.00005,
	l1_ratio=0.5,
).fit(TFIDFvectorised_dataset)
print(f"Done in {humanize.precisedelta(time.time() - start_time, suppress=['days', 'microseconds'])}")

print("\n", "Fitting the MiniBatchNMF model (generalized Kullback-Leibler divergence) with tf-idf")
KLMBNMF = MiniBatchNMF(
	n_components=n_components,
	random_state=1,
	batch_size=batch_size,
	init=init,
	beta_loss="kullback-leibler",
	alpha_W=0.00005,
	alpha_H=0.00005,
	l1_ratio=0.5,
).fit(TFIDFvectorised_dataset)
print(f"Done in {humanize.precisedelta(time.time() - start_time, suppress=['days', 'microseconds'])}")

tfidf_feature_names = TFIDFvectorizer.get_feature_names_out()

print("Plotting...")
n_top_words = 20
plot_top_words(
	FrobNMF,
	tfidf_feature_names,
	n_top_words,
	"Topics in NMF model (Frobenius norm)",
)

plot_top_words(
	KLNMF,
	tfidf_feature_names,
	n_top_words,
	"Topics in NMF model (generalized Kullback-Leibler divergence)",
)

plot_top_words(
    FrobMBNMF,
    tfidf_feature_names,
    n_top_words,
    "Topics in MiniBatchNMF model (Frobenius norm)",
)

plot_top_words(
    KLMBNMF,
    tfidf_feature_names,
    n_top_words,
    "Topics in MiniBatchNMF model (generalized Kullback-Leibler divergence)",
)