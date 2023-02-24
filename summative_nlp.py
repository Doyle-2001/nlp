#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import stem
stemmer = stem.PorterStemmer()
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
import string
punct = list(string.punctuation)
from collections import Counter
import requests
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().system('pip install PRAW')
import numpy as np
import praw
import datetime


# In[2]:


reddit = praw.Reddit(user_agent='VAD',
                     client_id='kleuv01oGlm-Ys0OYhJrpQ', client_secret="LzOpL1n8ZNdHLCLidI79-Aq43h5ROg",
                     username='SouthMobile8105', password='2KDRc,+rc+ZWQt$')


# here I am looking for 30 posts on the 'hot' tab of the Premier League subreddit
sub = reddit.subreddit("PremierLeague")
posts = sub.hot(limit=30)

# the code below sorts the posts by the number of comments
posts = sorted(posts, key=lambda post: post.num_comments, reverse=True)


# In[3]:


data = []


# In[4]:


# Iterate over the top 30 posts and their comments

# If a comment is an instance of 'MoreComments' it is skipped

for post in posts:
    comments = post.comments.list()
    for comment in comments:
        if isinstance(comment, praw.models.MoreComments):
            continue
        # Check if the author attribute is None
        author = comment.author.name if comment.author is not None else "[deleted]"
        # Convert the UTC timestamp to datetime object
        created_datetime = datetime.datetime.utcfromtimestamp(comment.created_utc)
        # Append the comment data to the list
        data.append({
            "comment_id": comment.id,
            "post_id": post.id,
            "author": author,
            "body": comment.body,
            "score": comment.score,
            "subreddit": 'r/PremierLeague',
            "title": post.title,
            "created_datetime": created_datetime,
            "num_replies": len(comment.replies),
        })


# In[5]:


# this converts the data collected into a pandas dataframe
df = pd.DataFrame(data)

# I've printed the first and last few rows to check how the data looks
print(df.head())
print(df.tail())


# In[6]:


df


# In[7]:


# Encode the 'body' column to remove non-ASCII characters
df["processed_body"] = df['body'].apply(lambda x: x.encode('ascii', 'ignore'))

# Decode the 'body' column to convert it back to a string
df['processed_body'] = df['processed_body'].apply(lambda x: x.decode())


# In[8]:


# turn all comments into lowercase for easy analysis
df['processed_body'].str.lower()


# In[9]:


# remove stopwords from the data 

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

df['processed_body'] = df['processed_body'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop_words]))


# In[10]:


# remove punctuation from the data
df['processed_body'] = df['processed_body'].str.replace('[{}]'.format(string.punctuation), '')


# In[11]:


# Tokenize and lemmatize the comments
lemmatizer = WordNetLemmatizer()
df["processed_body"] = df["processed_body"].apply(lambda x: " ".join([lemmatizer.lemmatize(w) for w in word_tokenize(x.lower())]))


# In[12]:


df


# In[13]:


# creating an empty list to store all the comments as individual words 
# (now the data has been cleaned) ready for sentiment analysis

words = []

for text in df['processed_body']:
    words += text.split()

print(words)


# In[14]:


# this is an existing excel file with VAD scores for corresponding words, 
# in order to run this code this excel file exists within my jupyter file
vad = pd.read_excel('vad.xlsx', index_col = 0)


# In[15]:


text = []
vad_score = []

for i in words:
    if i in vad.index:
        vad_score.append(vad.loc[i])
        # append the set of values that are in the index for that word
        text.append(i)
        # and then take the word and put it into the text list
    else:
        pass
    # and if a word doesnt satisfy this criteria, ignore it


# In[16]:


# this creates a dataframe to store the comments and their VAD score alongside

vad_df = pd.DataFrame(vad_score, index = text)


# In[17]:


vad_df


# In[18]:


get_ipython().system('pip install plotly')
import plotly.express as px


# In[19]:


# creating a visual to display the spread of data 
# the visual is interactive and will allow a user to see individual word scores 
# as well as any existing clusters of words

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters = 5).fit(vad)
# you can change the number of clusters with the line above ^^^
clusters = clustering.labels_
vad['clusters'] = clusters

fig = px.scatter_3d(vad_df, x='valence', y='arousal', z='dominance', hover_data = [vad_df.index])
fig.update_traces (marker=dict(size=5,
                              line=dict(width=2,
                                       color='DarkSlateGrey')),
                   selector=dict(mode='marker'))

fig.show("notebook")


# In[ ]:


# below I've created a displot for each VAD score - displaying the spread of the data across 
# each measurement of sentiment


# In[24]:


# Save the processed data to a CSV file
df.to_csv("reddit_comments.csv", index=False)

# this is now available within my github: https://github.com/Doyle-2001/nlp/tree/main


# In[29]:


import numpy as np 

# below I am comparing the median scores of the data collect from the sub reddit r/PremierLeague
# with the median scores of the data from the English Language


# In[36]:


np.median(vad['valence'])


# In[35]:


np.median(vad_df['valence'])


# In[39]:


np.median(vad['arousal'])


# In[40]:


np.median(vad_df['arousal'])


# In[30]:


np.median(vad['dominance'])


# In[32]:


np.median(vad_df['dominance'])


# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the density plot for 'dominance' column
sns.kdeplot(data=vad['valence'], label='Valence')

# Plot the density plot for 'valence' column
sns.kdeplot(data=vad_df['valence'], label='Valence')

# Set the title and axis labels
plt.title('Density plot of Valence Scores')
plt.xlabel('Score')
plt.ylabel('Density')

# Show the plot
plt.show()


# In[45]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the density plot for 'dominance' column
sns.kdeplot(data=vad['arousal'], label='Arousal')

# Plot the density plot for 'valence' column
sns.kdeplot(data=vad_df['arousal'], label='Arousal')

# Set the title and axis labels
plt.title('Density plot of Arousal Scores')
plt.xlabel('Score')
plt.ylabel('Density')

# Show the plot
plt.show()


# In[43]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the density plot for 'dominance' column
sns.kdeplot(data=vad['dominance'], label='Dominance')

# Plot the density plot for 'valence' column
sns.kdeplot(data=vad_df['dominance'], label='Domiance')

# Set the title and axis labels
plt.title('Density plot of Dominance Scores')
plt.xlabel('Score')
plt.ylabel('Density')

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




