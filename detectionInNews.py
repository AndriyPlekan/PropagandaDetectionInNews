#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.preprocessing import text,sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


fake_data = pd.read_csv('C:/University/fake_data.csv')
real_data = pd.read_csv('C:/University/real_data.csv')


# In[3]:


real_data.head()


# In[4]:


fake_data.head()


# In[5]:


real_data['target'] = 1
fake_data['target'] = 0 


# In[6]:


real_data.tail()


# In[7]:


data = pd.concat([real_data, fake_data], ignore_index=True, sort=False)
data.head()


# In[8]:


data.isnull().sum()


# In[9]:


print(data["target"].value_counts())
fig, ax = plt.subplots(1,2, figsize=(19, 5))
g1 = sns.countplot(data.target,ax=ax[0],palette="pastel");
g1.set_title("Розподіл даних")
g1.set_ylabel("Кількість")
g1.set_xlabel("Значення")
g2 = plt.pie(data["target"].value_counts().values,explode=[0,0],labels=data.target.value_counts().index, autopct='%1.1f%%',colors=['SkyBlue','PeachPuff'])
fig.show()


# In[10]:


print(data.subject.value_counts())
plt.figure(figsize=(10, 5))

ax = sns.countplot(x="Категорія", y="Кількість", hue='target', data=data, palette="pastel")
plt.title("Розподіл категорій згідно з реальними та неправдивими даними")


# In[11]:


data['text']= data['subject'] + " " + data['title'] + " " + data['text']
del data['title']
del data['subject']
del data['date']
data.head()


# In[44]:


first_text = data.text[10]
first_text


# In[45]:


from bs4 import BeautifulSoup

soup = BeautifulSoup(first_text, "html.parser")
first_text = soup.get_text()
first_text


# In[47]:


first_text = re.sub('\[[^]]*\]', ' ', first_text)
first_text = re.sub("[^a-zA-Zа-яА-ЯієїІЄЇІ]",' ',first_text)
first_text = first_text.lower()
first_text


# In[53]:


import nltk
from nltk.corpus import stopwords  

first_text = nltk.word_tokenize(first_text)


# In[54]:


first_text


# In[55]:


first_text = [ word for word in first_text if not word in set(stopwords.words("ukrainian"))]
first_text


# In[56]:


lemma = nltk.WordNetLemmatizer()
first_text = [ lemma.lemmatize(word) for word in first_text] 
first_text


# In[58]:


import pymorphy2
morph = pymorphy2.MorphAnalyzer(lang='uk')

b = [ morph.parse(word)[0].normal_form for word in first_text] 
b


# In[59]:


#Removal of HTML Contents
def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removal of Punctuation Marks
def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)

# Removal of Special Characters
def remove_characters(text):
    return re.sub("[^a-zA-Zа-яА-ЯієїІЄЇІ]"," ",text)

#Removal of stopwords 
def remove_stopwords_and_lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    for word in text:
        if word not in set(stopwords.words('ukrainian')):
            lemma = morph.parse(word)[0].normal_form
            final_text.append(lemma)
    return " ".join(final_text)

def cleaning(text):
    text = remove_html(text)
    text = remove_punctuations(text)
    text = remove_characters(text)
    text = remove_stopwords_and_lemmatization(text)
    return text

data['text']=data['text'].apply(cleaning)


# In[60]:


data.head(10)


# In[62]:


from wordcloud import WordCloud,STOPWORDS


# In[63]:


plt.figure(figsize = (15,15))
wc = WordCloud(max_words = 500 , width = 1000 , height = 500 , stopwords = stopwords.words("ukrainian")).generate(" ".join(data[data.target == 1].text))
plt.imshow(wc , interpolation = 'bilinear')


# In[65]:


plt.figure(figsize = (15,15))
wc = WordCloud(max_words = 500 , width = 1000 , height = 500 , stopwords = stopwords.words("ukrainian")).generate(" ".join(data[data.target == 0].text))
plt.imshow(wc , interpolation = 'bilinear')


# In[66]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len=data[data['target']==0]['text'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='SkyBlue')
ax1.set_title('Текст у неправдивих новинах')
text_len=data[data['target']==1]['text'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='PeachPuff')
ax2.set_title('Текст у реальних новинах')
plt.show()


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], random_state=0)


# In[31]:


max_features = 10000
maxlen = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data_train)
tokenized_train = tokenizer.texts_to_sequences(data_train)
data_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(data_test)
data_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)


# In[37]:


data_train


# In[39]:


batch_size = 256
epochs = 10
embed_size = 100

model = Sequential()
model.add(Embedding(max_features, output_dim=embed_size, input_length=maxlen, trainable=False))
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 32 , activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# In[40]:


history = model.fit(data_train, y_train, validation_split=0.3, epochs=10, batch_size=batch_size, shuffle=True, verbose = 1)


# In[41]:


print("Accuracy of the model on Training Data is - " , model.evaluate(data_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(data_test,y_test)[1]*100 , "%")


# In[42]:


plt.figure()
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("Точність")
plt.ylabel("Точність")
plt.xlabel("Епоха")
plt.legend()
plt.show()


# In[43]:


plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Похибка")
plt.ylabel("Точність")
plt.xlabel("Епоха")
plt.legend()
plt.show()


# In[44]:


predictions = model.predict_classes(data_test)
print(classification_report(results_test, predictions, target_names = ['Fake','Real']))


# In[46]:


results_test

