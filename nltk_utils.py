import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokanize(statement):
  return nltk.word_tokenize(statement)

def stem(word):
  #word = str(word)
  return stemmer.stem(str(word).lower())

def bag_of_words(tokenized_statement, all_words):
  tokenized_statement = [stem(w) for w in tokenized_statement]

  bag = np.zeros(len(all_words), dtype=np.float32)
  for i, w in enumerate(all_words):
    if w in tokenized_statement:
      bag[i] = 1.0
    
  return bag

# sentence = ["hello", "how", "are", "you"]
# words = ['hey', 'you', 'hello', 'how', 'hi', 'are', 'me']
# bag = bag_of_words(sentence, words)
# print(bag)

# s1 = "How are you? How was your day today?"
# print(s1)

# s1 = tokanize(s1)
# print(s1)

# s2 = ["organize", "organizing", "Organization", "organized", "organ"]
# stemmed_s2 = [stem(w) for w in s2]
# print(stemmed_s2)

# stemmed_s1 = [stem(w) for w in s1]
# print(stemmed_s1)