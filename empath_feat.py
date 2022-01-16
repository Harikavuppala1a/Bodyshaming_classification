
import os
from os import path
import csv
from wordcloud import WordCloud

text = ""
text_0 = ""
text_1=""
count = 0
count0 = 0
count1 = 0
from empath import Empath
lexicon = Empath()

with open("data/BS_data.csv", 'r') as csvfile:
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


pos_dict = lexicon.analyze(text_0, normalize=False)
neg_dict = lexicon.analyze(text_1, normalize=False)

diff = {key: pos_dict[key] - neg_dict.get(key, 0) for key in pos_dict}
sorted_diff = dict(sorted(diff.items(), key=lambda item: item[1], reverse= True))

print (sorted_diff)
