from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

twenty_train = fetch_20newsgroups(subset='train')
twenty_test = fetch_20newsgroups(subset='test')


text_classifier = Pipeline([
	('vectoriser', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('classifier', MultinomialNB()),
])
text_classifier.fit(twenty_train.data, twenty_train.target)

docs_test = twenty_test.data
predicted = text_classifier.predict(docs_test)
print(np.mean(predicted == twenty_test.target))


text_classifier = Pipeline([
	('vectoriser', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('classifier', SGDClassifier(loss='hinge', penalty='l2',
						  alpha=1e-3, random_state=42,
						  max_iter=5, tol=None)),
])

text_classifier.fit(twenty_train.data, twenty_train.target)
predicted = text_classifier.predict(docs_test)
print(np.mean(predicted == twenty_test.target))



parameters = {
	'vectoriser__ngram_range': [(1, 1), (1, 2)],
	'tfidf__use_idf': (True, False),
	'classifier__alpha': (1e-2, 1e-3),
}
gridsearch_classifier = GridSearchCV(text_classifier, parameters, n_jobs=-1)
gridsearch_classifier = gridsearch_classifier.fit(twenty_train.data, twenty_train.target)

disp = ConfusionMatrixDisplay.from_estimator(
		gridsearch_classifier,
		twenty_train.data,
		twenty_train.target,
		display_labels=twenty_train.target_names,
		cmap=plt.cm.Blues,
		normalize="true",
	)
plt.xticks(rotation=90)
plt.show()