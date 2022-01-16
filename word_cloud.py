import os

from os import path
import csv
from wordcloud import WordCloud
import string

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

text = ""
text_0 = ""
text_1=""
count = 0
count0 = 0
count1 = 0
cleaned_text0 = []
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

			tokens_0 = text_1.split()
			cleaned = [x.strip(string.punctuation) for x in tokens_0]

			comment_words = ".join(cleaned) "

wordcloud = WordCloud().generate(comment_words)
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40, background_color = 'white', colormap='Blues', collocations=True,max_words=50).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
plt.savefig('wordcloud_positive_cleaned_new.pdf', bbox_inches='tight')
