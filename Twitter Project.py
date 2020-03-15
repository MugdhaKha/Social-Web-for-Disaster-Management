
# coding: utf-8

# # Social Web for Disaster Management
# 
# Research has found that during natural calamities and disaster events, twitter is inundated with tweets some of which are tweets posted by victims (which we classify as "need" tweets). Organisations, volunteering camps and other relief camps also post tweets with regards to the resources and help they have to offer ( classified as "availability" tweets). 
# 
# Our project is to extract such tweets during disasters, classify them as need and availability tweets and map the need tweets to the respective availability tweets to expedite relief efforts.
# 
# For example, need tweets addressing requirement of water, medical aid etc.. should be mapped to availability tweets offering help for the same.
# 
# Clustering has been performed with the intent to cluster the tweets into need and availability. ( That objective hasn't been met yet though)
# 
# -Retrieved tweets using twitter API.
# 
# -Dataset being used pertains to the 2015 Nepal earthquake.
# 
# Implemented the following in this notebook : 
# 
# 1) Converted tweets to numerical using gensim's doc2vec model.
# 
# 2) Dimensionality reduction using tSNE.
# 
# 3) Performed k-means clustering
# 
# 4) Performed DBSCAN clustering

# In[2]:


#All imports
import gensim
import os
import collections
import smart_open
import random
import sklearn.manifold
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd


# In[5]:


# Set file names for train and test data
filename = 'dataset_project.txt'


# In[3]:


def read_corpus(fname):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


# In[6]:


index = []
tweet = []
with smart_open.smart_open(filename, encoding="iso-8859-1") as f:
    for i, line in enumerate(f):
        #print(i, ":" , line, "\n")
        index.append(i)
        tweet.append(line)


# In[7]:


train_corpus = list(read_corpus(filename))


# In[9]:


train_corpus[:2]


# # Doc2vec

# In[10]:


model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=50)


# In[11]:


model.build_vocab(train_corpus)


# In[12]:


get_ipython().run_line_magic('time', 'model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)')


# # tSNE

# In[13]:


get_ipython().run_line_magic('time', 'tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)')


# In[14]:


get_ipython().run_line_magic('time', 'tsne_d2v = tsne.fit_transform(model.docvecs.vectors_docs)')


# In[15]:


tsne_d2v_df = pd.DataFrame(data=tsne_d2v, columns=["x", "y"])


# In[16]:


tsne_d2v_df['Tweet'] = tweet


# In[17]:


tsne_d2v_df


# In[18]:


tsne_d2v_df.plot.scatter("x", "y", s=10, figsize=(20, 12))


# In[19]:


Y = list(zip(tsne_d2v_df['x'],tsne_d2v_df['y']))


# # K means clustering

# In[21]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(Y)
#create a dictionary to get cluster data
# clusters={0:[],1:[],2:[],3:[]}
# for i in range(40):
#     clusters[kmeans.labels_[i]].append(' '.join(df_new.ix[i,'title_l']))
# print(clusters)


# In[22]:


fig = plt.figure(figsize=(15, 10))
# plt.rcParams["figure.figsize"] = [15,10]
ax = fig.add_subplot(1, 1, 1)
c = kmeans.labels_
ix = np.where(c==1)
ax.plot(Y[ix,0], Y[ix,1], 'o',markerfacecolor='red', markersize=12)
ix = np.where(c==0)
ax.plot(Y[ix,0], Y[ix,1], 'o',markerfacecolor='green', markersize=12)
plt.show()


# In[23]:


(list(kmeans.labels_)).count(1)


# In[24]:


(list(kmeans.labels_)).count(0)


# In[ ]:


grp = list(zip(index,tweet,list(kmeans.labels_)))


# In[ ]:




df = pd.DataFrame(grp, columns=['Index','Tweet','Cluster'])


# # DBSCAN Clustering

# In[83]:


Y = StandardScaler().fit_transform(Y)
#db = DBSCAN(eps=0.05, min_samples=30).fit(Y) - 162
#db = DBSCAN(eps=0.05, min_samples=50).fit(Y) - 139
#db = DBSCAN(eps=0.05, min_samples=40).fit(Y) - 201 
#db = DBSCAN(eps=0.047, min_samples=40).fit(Y) - 184 
#db = DBSCAN(eps=0.047, min_samples=35).fit(Y) - 221
#db = DBSCAN(eps=0.04, min_samples=26).fit(Y) - 295
#db = DBSCAN(eps=0.06, min_samples=26).fit(Y) - 78
db = DBSCAN(eps=0.06, min_samples=26).fit(Y)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


# In[84]:


print(n_clusters_)


# In[85]:


unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    plt.rcParams["figure.figsize"] = [15,10]
    class_member_mask = (labels == k)
    xy = Y[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7)
        
    xy = Y[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)
    
# plt.figure(figsize=(20,12))
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# In[ ]:


grp2 = list(zip(index,tweet,list(labels)))


# In[ ]:


db_df = pd.DataFrame(grp2, columns= ['Index','Tweet','Cluster'])

