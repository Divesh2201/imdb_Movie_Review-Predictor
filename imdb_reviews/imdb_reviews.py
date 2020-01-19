import tensorflow as tf
from tensorflow import keras
import numpy as np 
 
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000)
# Here num_words = 10,000 essentially means that we have a vocabulary of 10000 words only. 
# This also shrinks our data which makes it a little bit nicer :)

word_index = data.get_word_index()
# This gives us tuples which contain the strings of the words corresponding to the integers.

word_index = {k:(v+4) for k, v in word_index.items()}
# This just breaks the tuple into key and value pairs.

word_index["<PAD>"] = 0
# Here padding is used to make each movie reivew of the same length.
word_index["<START>"] = 1
word_index["<UNK>"] = 2
# This is for the unknown characters that are not in the dictionary.
word_index["<UNUSED>"] = 3
word_index["<BREAK>"] = 4

reverse_word_index = dict([(key, value) for (value, key) in word_index.items()])
# This dictinary reverses the word_index, now the integers point to the word.


def decode_review(text):
	return " ".join([reverse_word_index.get(i,"?") for i in text])

# This function decodes the review from integers to words that are present in the dictionary.

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 270)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"], padding = "post", maxlen = 270)

# This inbuilt keras function does the padding for train and test data.

# Model down here

model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = "relu"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))

# The sigmoid activation layer is the output layer (hypothesis) of the neural network which consists of
# one neuron which gives out the prediction as 0 or 1.
# Moreover in this layer the sigmoid function manages the probablities and returns the desired output.

# Some Intuition behind the embedding layer...
# The Embedding layer tries to group the words that are similar to each other.
# Mathematically, the Embedding layer finds word vectors for each word that we pass it.
# A word vector can be any dimensional space.Now here we've picked 16 dimensions for each word vector.
# Initially, we create 10000 word vectors for every single word.
# When we call the embedding layer...it grabs all of those word vectors for whatever input we have
# and use that as the data that we pass on to the next layer.
# So it looks at the context the words have been used for and groups similar words together.

# The output of the embedding layer gives us a 16 dimension vector(16 coefficients).
# The GlobalAveragePooling1D() essentially just scales down the data (the dimensions) cuz we have tons 
# of words and each word gives a 16 dimension output.

model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs = 40, batch_size = 500, validation_data = (x_val, y_val), verbose = 1)

results = model.evaluate(test_data, test_labels)

model.save("imdb_reviews_model.h5")

print(results)

test_review = test_data[15]
predict = model.predict([test_review])
print("Review for test_data")
print(decode_review(test_review))
print("Prediction"+ str(predict[15]))
print("Actual"+ str(test_labels[15]))
print(results)




