import pandas as pd
from googletrans import Translator
import time
from scipy import stats
from scipy.stats import ttest_ind
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import hstack

analyzer = SentimentIntensityAnalyzer()

translator = Translator()


df = pd.read_csv('/Users/gpasion/Documents/INST414/combined_output.csv')


#duplicates = df.duplicated()
#print(duplicates)  # Shows True for each duplicated row

#num_duplicates = df.duplicated().sum()
#print(f"Total duplicate rows: {num_duplicates}")

# df['translated'] = df['text'].apply(lambda x: translator.translate(str(x), src='tl', dest='en').text)

#sample = df['text'].head(10)  # Only translate first 10 rows
#df['translated'] = sample.apply(lambda x: translator.translate(str(x), src='tl', dest='en').text)


#print(df[['text', 'translated']])

#df.to_csv("/Users/gpasion/Documents/INST414/translatedsample.csv", index=False)


# def safe_translate_with_delay(text):
#     try:
#         time.sleep(1)  # 1-second delay
#         return translator.translate(str(text), src='tl', dest='en').text
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# df['translated'] = df['text'].apply(safe_translate_with_delay)

# df.to_csv("/Users/gpasion/Documents/INST414/translated.csv", index=False)


# relevancy adds the column if the twitter posts were relevant to the research topic
df_relevancy = pd.read_csv('/Users/gpasion/Documents/INST414/relevancy.csv')

true_count = df_relevancy["Relevant"].sum()

print("Relevant: ", true_count)

df_phaseout = pd.read_csv("/Users/gpasion/Documents/INST414/phaseout.csv")


# where phaseout is true
po_true = df_phaseout[df_phaseout["phaseout"] == True]
# where phaseout is false
po_false = df_phaseout[df_phaseout["phaseout"] == False]
yes_po = df_phaseout["phaseout"].sum()
no_po = (df_phaseout["phaseout"] == False).sum()
na_po = df_phaseout["phaseout"].isna().sum()
relevant_total_po = yes_po + no_po

print("Yes to phaseout: ", yes_po)
print("No to phaseout: ", no_po)
print("Irrelevant: ", na_po)
print("Total relevant posts: ", (yes_po + no_po))

# descriptive stats: yes to phaseout

# percentage of yes to phaseout
percent_yes_po = yes_po / relevant_total_po
print(percent_yes_po)
# percentage of no to phaseout
percent_no_po = no_po / relevant_total_po
print(percent_no_po)

#likes

# descriptive stats: no to phaseout (likes)

like_count_stats_yespo = po_true['likeCount'].describe()
print(like_count_stats_yespo)

like_count_stats_nopo = po_false['likeCount'].describe()
print(like_count_stats_nopo)

# two sample t test

t_stat, p_val = ttest_ind(po_true['likeCount'], po_false['likeCount'], equal_var=False)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")

# Drop rows where 'phaseout' is NaN
df_filtered = df_phaseout[df_phaseout['phaseout'].notna()]


def get_sentiment_label(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment labeling
df_filtered['sentiment'] = df_filtered['translated'].apply(get_sentiment_label)
# apply sentiment score
df_filtered['sentiment_score'] = df_filtered['translated'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

df_filtered.to_csv('/Users/gpasion/Documents/INST414/sentiments.csv', index=False)


print(df_filtered.describe())

df_filtered.describe().to_csv('/Users/gpasion/Documents/INST414/phaseout_sentiments_descriptive_stats.csv', index=False)




## this is a maybe depending on capacity

# have a modernization column

# average posts: yes to modernization

# average posts: no to modernization

# average likes: yes to modernize

# average likes: no to modernize