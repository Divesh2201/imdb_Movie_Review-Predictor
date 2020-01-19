# We are creating a function which adds padding to the review if its shorter than maxlen words and cut the review greater than 
# maxlen words at that point.
def padding_review(data, value, maxlen):
	for item in data:
		if len(data[item])>=maxlen:
			del item[maxlen+1:]
		else:
			l = len(data[item])
			data[item[l:maxlen]] = value
	return data	 


train_data = padding_review(train_data, value = word_index["<PAD>"], maxlen = 270)
test_data = padding_review(test_data, value = word_index["<PAD>"], maxlen = 270)