import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
import keras
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(0, len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    allowed_chars = string.ascii_lowercase + ' ' + '!' + ',' + '.' + ':' + ';' + '?'

    # remove as many non-english characters and character sequences as you can
    for char in text:
        if char not in allowed_chars:
            text = text.replace(char, ' ')

    # shorten any extra dead space created above
    text = text.replace('  ', ' ')

    chars = sorted(list(set(text)))
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('-', ' ')
    text = text.replace('*', ' ')
    text = text.replace('`', ' ')
    text = text.replace('/', ' ')
    text = text.replace('&', ' ')
    text = text.replace('%', ' ')
    text = text.replace('@', ' ')
    text = text.replace('$', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('{', ' ')
    text = text.replace('}', ' ')
    text = text.replace('=', ' ')
    text = text.replace('+', ' ')
    text = text.replace('_', ' ')
    text = text.replace('"', ' ')
    text = text.replace("'", ' ')
    text = text.replace('   ', ' ')
    text = text.replace('^', ' ')
    text = text.replace('~', ' ')
    text = text.replace('<', ' ')
    text = text.replace('>', ' ')
    text = text.replace('|', ' ')
    text = text.replace('1', ' ')
    text = text.replace('2', ' ')
    text = text.replace('3', ' ')
    text = text.replace('4', ' ')
    text = text.replace('5', ' ')
    text = text.replace('6', ' ')
    text = text.replace('7', ' ')
    text = text.replace('8', ' ')
    text = text.replace('9', ' ')
    text = text.replace('0', ' ')

    # shorten any extra dead space created above
    text = text.replace('  ', ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation("softmax"))
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
