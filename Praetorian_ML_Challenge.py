
# coding: utf-8

# In[171]:

import pandas as pd

df = pd.read_csv("binary_data.csv")

df.head()


# In[172]:

df.info()


# In[232]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

X = df.binary
y = df["class"]
vec_opts = {
    "ngram_range": (2, 3),  # allow n-grams of 1-4 words in length (32-bits)
    "analyzer": "word",     # analyze hex words
    "token_pattern": "..",  # treat two characters as a word (e.g. 4b)
    "min_df" : 0.001
}
v = CountVectorizer(**vec_opts)


# In[233]:

from sklearn.feature_extraction.text import TfidfTransformer

idf_opts = {"use_idf": True}
idf = TfidfTransformer(**idf_opts)


# In[234]:

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('vec',   CountVectorizer(**vec_opts)),
    ('idf',  TfidfTransformer(**idf_opts)),
])

X = pipeline.fit_transform(X, y)


# In[235]:

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

clf = MultinomialNB().fit(X, y)
all_predictions = clf.predict(X)


# In[236]:

from sklearn.metrics import classification_report
print classification_report(y, all_predictions)


# In[237]:

import requests
import logging
import base64
import time

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Server(object):
    url = 'https://mlb.praetorian.com'
    log = logging.getLogger(__name__)

    def __init__(self):
        self.session = requests.session()
        self.binary  = None
        self.hash    = None
        self.wins    = 0
        self.targets = []

    def _request(self, route, method='get', data=None):
        while True:
            try:
                if method == 'get':
                    r = self.session.get(self.url + route)
                else:
                    r = self.session.post(self.url + route, data=data)
                if r.status_code == 429:
                    raise Exception('Rate Limit Exception')
                if r.status_code == 500:
                    raise Exception('Unknown Server Exception')

                return r.json()
            except Exception as e:
                self.log.error(e)
                self.log.info('Waiting 60 seconds before next request')
                time.sleep(60)

    def get(self):
        r = self._request("/challenge")
        self.targets = r.get('target', [])
        self.binary  = base64.b64decode(r.get('binary', ''))
        return r

    def post(self, target):
        r = self._request("/solve", method="post", data={"target": target})
        self.wins = r.get('correct', 0)
        self.hash = r.get('hash', self.hash)
        self.ans  = r.get('target', 'unknown')
        return r

if __name__ == "__main__":
    import random
    s = Server()
    var = 1
    while var == 1:
        # query the /challenge endpoint
        s.get()
        # query the /challenge endpoint        
        #print binascii.hexlify(s.binary)
        
        test_binary = binascii.hexlify(s.binary)
       
        X_test = pipeline.transform([test_binary])

        test_prediction = clf.predict(X_test)
        
        s.post(test_prediction)
        
        s.log.info("Guess:[{: >9}]   Answer:[{: >9}]   Wins:[{: >3}]".format(test_prediction, s.ans, s.wins))

        # 500 consecutive correct answers are required to win
        # very very unlikely with current code
        if s.hash:
            s.log.info("You win! {}".format(s.hash))


# In[ ]:



