
# coding: utf-8

# In[18]:

import nltk
from collections import Counter
import pandas as pd
import string
import numpy as np
import sklearn

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')

# # Text Classification [30pts]
# In this problem (again!), you will be analyzing the Twitter data we extracted using [this](https://dev.twitter.com/overview/api) api. This time, we extracted the tweets posted by the following six Twitter accounts: `realDonaldTrump, mike_pence, GOP, HillaryClinton, timkaine, TheDemocrats`
# 
# For every tweet, we collected two pieces of information:
# - `screen_name`: the Twitter handle of the user tweeting and
# - `text`: the content of the tweet.
# 
# We divided the tweets into three parts - train, test and hidden test - the first two of which are available to you in CSV files. For train, both the `screen_name` and `text` attributes were provided but for test, `screen_name` was hidden.
# 
# The overarching goal of the problem is to "predict" the political inclination (Republican/Democratic) of the Twitter user from one of his/her tweets. The ground truth (i.e., true class labels) is determined from the `screen_name` of the tweet as follows
# - `realDonaldTrump, mike_pence, GOP` are Republicans
# - `HillaryClinton, timkaine, TheDemocrats` are Democrats
# 
# Thus, this is a binary classification problem. 
# 
# The problem proceeds in three stages:
# 1. **Text processing (8pts)**: We will clean up the raw tweet text using the various functions offered by the [nltk](http://www.nltk.org/genindex.html) package.
# 2. **Feature construction (10pts)**: In this part, we will construct bag-of-words feature vectors and training labels from the processed text of tweets and the `screen_name` columns respectively.
# 3. **Classification (12pts)**: Using the features derived, we will use [sklearn](http://scikit-learn.org/stable/modules/classes.html) package to learn a model which classifies the tweets as desired. 
# 
# As mentioned earlier, you will use two new python packages in this problem: `nltk` and `sklearn`, both of which should be available with anaconda. However, NLTK comes with many corpora, toy grammars, trained models, etc, which have to be downloaded manually. This assignment requires NLTK's stopwords list and WordNetLemmatizer. Install them using:
# 
#   ```python
#   >>>nltk.download('stopwords')
#   >>>nltk.download('wordnet')
#   ```
# 
# Verify that the following commands work for you, before moving on.
# 
#   ```python
#   >>>lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()
#   >>>stopwords=nltk.corpus.stopwords.words('english')
#   ```
# 
# Let's begin!

# ## 1. Text Processing [6pts + 2pts]
# 
# You first task to fill in the following function which processes and tokenizes raw text. The generated list of tokens should meet the following specifications:
# 1. The tokens must all be in lower case.
# 2. The tokens should appear in the same order as in the raw text.
# 3. The tokens must be in their lemmatized form. If a word cannot be lemmatized (i.e, you get an exception), simply catch it and ignore it. These words will not appear in the token list.
# 4. The tokens must not contain any punctuations. Punctuations should be handled as follows: (a) Apostrophe of the form `'s` must be ignored. e.g., `She's` becomes `she`. (b) Other apostrophes should be omitted. e.g, `don't` becomes `dont`. (c) Words must be broken at the hyphen and other punctuations. 
# 
# Part of your work is to figure out a logical order to carry out the above operations. You may find `string.punctuation` useful, to get hold of all punctuation symbols. Your tokens must be of type `str`. Use `nltk.word_tokenize()` for tokenization once you have handled punctuation in the manner specified above.

# In[2]:

def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    text = text.lower()
    res = text.replace("'s", "")
    res = res.replace("'", "")
    new = ""
    for i in range(len(res)):
        if(res[i] in string.punctuation):
            new += " "
        else:
            new += res[i]
    thing = nltk.word_tokenize(new)
    result = []
    for word in thing:
        try:
            result.append(lemmatizer.lemmatize(word))
        except:
            pass
    return result


# You can test the above function as follows. Try to make your test strings as exhaustive as possible. Some checks are:
#     
#    ```python
#    >>> process("I'm doing well! How about you?")
#    ['im', 'doing', 'well', 'how', 'about', 'you']
#    ```
# 
#    ```python
#    >>> process("Education is the ability to listen to almost anything without losing your temper or your self-confidence.")
#    ['education', 'is', 'the', 'ability', 'to', 'listen', 'to', 'almost', 'anything', 'without', 'losing', 'your', 'temper', 'or', 'your', 'self', 'confidence']
#    ```

# In[3]:



# You will now use the `process()` function we implemented to convert the pandas dataframe we just loaded from tweets_train.csv file. Your function should be able to handle any data frame which contains a column called `text`. The data frame you return should replace every string in `text` with the result of `process()` and retain all other columns as such. Do not change the order of rows/columns.

# In[6]:

def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process_text() function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    cpy = df.copy()
    #i = 0
    for thing in cpy.iterrows():
        row = thing[1]
        txt = row['text']
        #if(i == 0):
            #print type(txt)
        row['text'] = process(txt, lemmatizer)
        
        #if(i == 0):
            
            #print type(row['text'])
        #i += 1
    return cpy

# The output should be:
# 
#    ```python
#     >>> print processed_tweets.head()
#           screen_name                                               text
#     0             GOP  [rt, gopconvention, oregon, vote, today, that,...
#     1    TheDemocrats  [rt, dwstweets, the, choice, for, 2016, is, cl...
#     2  HillaryClinton  [trump, calling, for, trillion, dollar, tax, c...
#     3  HillaryClinton  [timkaine, guiding, principle, the, belief, th...
#     4        timkaine  [glad, the, senate, could, pas, a, thud, milco...
#     ```

# ## 2. Feature Construction [4pts + 4pts + 2pts]
# The next step is to derive feature vectors from the tokenized tweets. In this section, you will be constructing a bag-of-words TF-IDF feature vector.
# 
# But before that, as you may have guessed, the number of possible words is prohibitively large and not all of them may be useful for our classification task. Our first sub-task is to determine which words to retain, and which to omit. The common heuristic is to construct a frequency distribution of words in the corpus and prune out the head and tail of the distribution. The intuition of the above operation is as follows. Very common words (i.e. stopwords) add almost no information regarding similarity of two pieces of text. Conversely, very rare words tend to be typos. 
# 
# As NLTK has a list of in-built stop words which is a good substitute for head of the distribution, we will now implement a function which identifies rare words (tail). We will consider a word rare if it occurs not more than once in whole of tweets_train.csv.
# 
# Using `collections.Counter` will make your life easier.
#    ```python
#    >>> Counter(['sample', 'test', 'input', 'processing', 'sample'])
#     Counter({'input': 1, 'processing': 1, 'sample': 2, 'test': 1})
#    ```
# For details on other operations you can perform with Counter, see [this](https://docs.python.org/2/library/collections.html#collections.Counter) page.

# In[7]:

def get_rare_words(processed_tweets):
    """ use the word count information across all tweets in training data to come up with a feature list
    Inputs:
        processed_tweets: pd.DataFrame: the output of process_all() function
    Outputs:
        list(str): list of rare words, sorted alphabetically.
    """
    allWords = []
    cpy = processed_tweets.copy()
    for thing in cpy.iterrows():
        row = thing[1]
        txt = row['text']
        allWords.extend(txt)
    counterObj = Counter(allWords)
    result = []
    for key in counterObj:
        if(counterObj[key] == 1):
            result.append(key)
    return sorted(result)


def create_features(processed_tweets, rare_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        tweets: pd.DataFrame: tweets read from train/test csv file, containing the column 'text'
        rare_words: list(str): one of the outputs of get_feature_and_rare_words() function
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
                                                we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    cpy = processed_tweets.copy()
    bbj = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = rare_words + nltk.corpus.stopwords.words('english'))
    allWords = []
    for thing in cpy.iterrows():
        row = thing[1]
        words = row['text']
        allWords.append(" ".join(words))
    return (bbj, bbj.fit_transform(allWords))



def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    res = np.empty(shape = len(processed_tweets), dtype = int)
    cpy = processed_tweets.copy()
    i = 0
    for thing in cpy.iterrows():
        row = thing[1]
        if(row['screen_name'] in ['realDonaldTrump', 'mike_pence', 'GOP']):
            res[i] = 0
        else:
            res[i] = 1
        i += 1
    return res
    

def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.classes.SVC: classifier learnt
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    cpy = process_all(unlabeled_tweets)
    allWords = []
    for thing in cpy.iterrows():
        row = thing[1]
        words = row['text']
        allWords.append(" ".join(words))
    return classifier.predict(tfidf.transform(allWords))

