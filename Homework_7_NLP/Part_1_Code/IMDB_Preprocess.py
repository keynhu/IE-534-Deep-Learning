# Import modules
import numpy as np
import itertools
import nltk
import datetime
import os
import shutil
import io

start_time = datetime.datetime.now()
print("Start Time: ", start_time)

## get all of the training reviews (including unlabeled reviews)
train_directory = '/projects/training/bauh/NLP/aclImdb/train/'
pos_filenames = os.listdir(train_directory + 'pos/')
neg_filenames = os.listdir(train_directory + 'neg/')
unsup_filenames = os.listdir(train_directory + 'unsup/')

pos_filenames = [train_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [train_directory+'neg/'+filename for filename in neg_filenames]
unsup_filenames = [train_directory+'unsup/'+filename for filename in unsup_filenames]

filenames = pos_filenames + neg_filenames + unsup_filenames

x_train = []
for i,filename in enumerate(filenames):
    with io.open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    x_train.append(line)
    if (i+1)%1000==0:
        print("Current line: ", i+1)
        now_time = datetime.datetime.now()
        print("Cost Time: ", now_time-start_time)
print("Train lines read!")
now_time = datetime.datetime.now()
print("Cost Time: ", now_time-start_time)
    
## get all of the test reviews
test_directory = '/projects/training/bauh/NLP/aclImdb/test/'

pos_filenames = os.listdir(test_directory + 'pos/')
neg_filenames = os.listdir(test_directory + 'neg/')

pos_filenames = [test_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [test_directory+'neg/'+filename for filename in neg_filenames]

filenames = pos_filenames+neg_filenames

x_test = []
for i, filename in enumerate(filenames):
    with io.open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    x_test.append(line)
    if (i+1)%1000==0:
        print("Current line: ", i+1)
        now_time = datetime.datetime.now()
        print("Cost Time: ", now_time-start_time)
print("Test lines read!")
now_time = datetime.datetime.now()
print("Cost Time: ", now_time-start_time)
    
## word_to_id and id_to_word. associate an id to every unique token in the training data
## Hence we can build our own vocabulary
all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token:idx for idx,token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)
## Sort the indices by word frequency instead of random
x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]
## recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}
## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
print("Train IDs created!")
now_time = datetime.datetime.now()
print("Cost Time: ", now_time-start_time)
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]
print("Test IDs created!")
now_time = datetime.datetime.now()
print("Cost Time: ", now_time-start_time)

## save dictionary
np.save('HW7/preprocessed_data/imdb_dictionary.npy',np.asarray(id_to_word))
## save training data to single text file
with io.open('HW7/preprocessed_data/imdb_train.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
## save test data to single text file
with io.open('HW7/preprocessed_data/imdb_test.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
print("Dictionary saved!")
now_time = datetime.datetime.now()
print("Cost Time: ", now_time-start_time)
        
## Read GloVe embeddings
glove_filename = '/projects/training/bauh/NLP/glove.840B.300d.txt'
with io.open(glove_filename,'r',encoding='utf-8') as f:
    lines = f.readlines()

glove_dictionary = []
glove_embeddings = []
count = 0
for line in lines:
    line = line.strip()
    line = line.split(' ')
    glove_dictionary.append(line[0])
    embedding = np.asarray(line[1:],dtype=np.float)
    glove_embeddings.append(embedding)
    count+=1
    if(count>=100000):
        break
print("GloVe lines saved!")
now_time = datetime.datetime.now()
print("Cost Time: ", now_time-start_time)

glove_dictionary = np.asarray(glove_dictionary)
glove_embeddings = np.asarray(glove_embeddings)
# added a vector of zeros for the unknown tokens
glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))

word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]
print("GloVe Ids created!")
now_time = datetime.datetime.now()
print("Cost Time: ", now_time-start_time)

np.save('HW7/preprocessed_data/glove_dictionary.npy',glove_dictionary)
np.save('HW7/preprocessed_data/glove_embeddings.npy',glove_embeddings)

with io.open('HW7/preprocessed_data/imdb_train_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
with io.open('HW7/preprocessed_data/imdb_test_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
print("GloVe Dictionary saved!")
now_time = datetime.datetime.now()
print("Cost Time: ", now_time-start_time)
