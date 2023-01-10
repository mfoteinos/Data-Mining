import numpy as np
import pandas as pd
from gensim.models import Word2Vec, word2vec
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import re
import random



def PreProcess(review):

    temp = []
    text = re.sub("[^a-zA-Z]", " ", review) # Remove non-letters
    text = text.lower().split() # Convert to words to lower case and split sentences into words

    stops = set(stopwords.words("english")) # Remove stopwords
    text = [w for w in text if not w in stops]

    temp.append(text)

    return temp


def Word2vec(text):

    model = word2vec.Word2Vec(text, workers=4, vector_size=100, min_count=40, window=10, sample=1e-3)
    model.save("word2vec.model")



def Get_Avg_Vec(text,model):

    avg_vec = np.zeros((100,), dtype='float32')
    in2word = set(model.wv.index_to_key)
    wr = 0

    for word in text:
        if word in in2word: # Finds if the words exists in the model of vectors
            wr += 1
            avg_vec = np.add(avg_vec, model.wv[word])   # For each sentence adds the vectors of each word and divides by the number of words

    avg_vec = np.divide(avg_vec, wr)

    return avg_vec # Returns the average vector of each sentence




def Avg_Vec(text , model):

    review_vecs = np.zeros((len(text), 100), dtype='float32')
    count = 0

    for review in text:
        review_vecs[count] = Get_Avg_Vec(review,model) # Finds the average vector of each review and adds it in the review_vecs list
        count +=1

    return review_vecs







df = pd.read_csv('amazon.csv')

text = []
x = 0


df = df.sample(frac=1).reset_index()    # Randomize the csv file
df = df.drop(columns='index')   # Remove previous index


for review in df['Text']:   # For every review in the csv remove stopwords, convert to lower case and split reviews into words, remove non-letters,
    text += PreProcess(review)
    x += 1
    print(x)


Word2vec(text)  # Give a unique vector to each word

model = Word2Vec.load("word2vec.model")     # Load the model of the vectors


train_text = []
test_text = []
train_text_score = []


x = int(len(text) * 0.8)
y = int(len(text))



for index in range(x):
    train_text.append(text[index])
                                        # Split reviews into train_text
train_text_score = df.iloc[0:x, 1]

train_text_vecs = Avg_Vec(train_text, model) # Find the average vector of each sentence in the train_text



for index in range(x, y):
    test_text.append(text[index])
                                        # Split reviews into test_text
test_text_score = df.iloc[x:y, 1]

test_text_vecs = Avg_Vec(test_text, model) # Find the average vector of each sentence in the test_text




nan_indices = list({k for k, l in np.argwhere(np.isnan(train_text_vecs))})  # Finds the index where the vector contains NaN
nan_indices.sort(reverse=True)
train_text_vecs = np.delete(train_text_vecs, nan_indices, axis=0) # From the train text vecs remove the vecs that contain NaN values and the corresponding train text score
for x in nan_indices:
    train_text_score.pop(x)


nan_indices = list({k for k, l in np.argwhere(np.isnan(test_text_vecs))})
nan_indices.sort(reverse=True)
test_text_vecs = np.delete(test_text_vecs, nan_indices, axis=0) # From the test text vecs remove the vecs that contain NaN values and the corresponding test text score
for x in nan_indices:
    test_text_score.pop(x+40000)


forest = RandomForestClassifier(n_estimators=1000)       # Fit a Random Forest to the train text using 1000 trees
forest.fit(train_text_vecs, train_text_score)
prediction = forest.predict(test_text_vecs)     # Find the prediction for the test text


f_score = f1_score(test_text_score, prediction, average='macro')
prec_score = precision_score(test_text_score, prediction, average='macro')
rec_score = recall_score(test_text_score, prediction, average='macro')


print("F1_Score:", f_score)
print("Precision_Score:", prec_score)
print("Recall_Score:", rec_score)








