import os
from os import path
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns

text = ""
text_0 = ""
text_1=""
count = 0
count0 = 0
count1 = 0
label =[]
tweet = []

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


with open("../data/BS_data.csv", 'r') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			post = str(row['text'])
			count = count + 1
			if row['label'] == '1':
				text_1 = text_1 + post
				count1= count1+1
			else:
				text_0 = text_0 + post
				count0 = count0 + 1
			text = text + post
			label.append(row['label'])
			tweet.append(row['text'])

# print (text_0)
tfidf_transformer = TfidfTransformer(norm = 'l2')
count_vec = CountVectorizer(analyzer="char",max_features = 5000, ngram_range = (1,5))
bow_transformer_train = count_vec.fit_transform(tweet)
# bow_transformer_test =count_vec.transform(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
train_features = tfidf_transformer.fit_transform(bow_transformer_train)
# test_features= tfidf_transformer.transform(bow_transformer_test )
print (train_features.shape)

standarized_data = StandardScaler().fit_transform(train_features.toarray())

model = TSNE(n_components=2, random_state=0,perplexity=10, n_iter=1000)
tsne_data = model.fit_transform(standarized_data)
# print (tsne_data)
df = pd.DataFrame()
df["y"] = label
df["comp-1"] = tsne_data[:,0]
df["comp-2"] = tsne_data[:,1]

print (df)
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="Body shaming data T-SNE projection")
plt.savefig('tsne_fig_per10_feat5000_niter1000.pdf', bbox_inches='tight')