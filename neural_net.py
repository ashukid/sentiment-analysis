# coding: utf-8
import pandas as pd
import numpy as np
import re
import tensorflow as tf

# manually defining the stopwords
stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are","as","at","be","because","been","before","being","below","between","both","but","by","could","did","do","does","doing","down","during","each","few","for","from","further","had","has","have","having","he","he’d","he’ll","he’s","her","here","here’s","hers","herself","him","himself","his","how","how’s","I","I’d","I’ll","I’m","I’ve","if","in","into","is","it","it’s","its","itself","let’s","me","more","most","my","myself","nor","of","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","she","she’d","she’ll","she’s","should","so","some","such","than","that","that’s","the","their","theirs","them","themselves","then","there","there’s","these","they","they’d","they’ll","they’re","they’ve","this","those","through","to","too","under","until","up","very","was","we","we’d","we’ll","we’re","we’ve","were","what","what’s","when","when’s","where","where’s","which","while","who","who’s","whom","why","why’s","with","would","you","you’d","you’ll","you’re","you’ve","your","yours","yourself","yourselves"]
number_of_classes = 2 # love and sad

# data preprocessing part
def data_preprocessing(train):
    x_train = train.loc[:,['Document']]
    y_train = train.loc[:,['Sentiment']]
    y_train = y_train['Sentiment'].astype('category')
    y_train.cat.categories = range(number_of_classes)
    y_train = np.ravel(y_train)
    x_train = list(x_train.Document)
    
    return (x_train,y_train)

def create_corpus(data):
    words_corpus = set()
    for row in data:
        row = re.sub(r'^[a-z][A-z]', '', row)
        row = row.split()
        row = [word for word in row if not word in stop_words]
        for word in row:
            words_corpus.add(word)
    
    words_corpus = list(words_corpus)
    return words_corpus

def count_vectorize(data,words_corpus):
    
    word_count = [[0]*len(words_corpus) for i in range(len(data))]
    i=0
    for row in data:
        row = re.sub(r'^[a-z][A-z]', '', row)
        row = row.split()
        row = [word for word in row if not word in stop_words]
        for word in row:
            if not word in stop_words:
                word_count[i][words_corpus.index(word)] += 1
        i+=1
  
    return word_count

def one_hot(data):
    one_hot_data = []
    for x in data:
        temp = list(np.zeros(number_of_classes))
        temp[x]=1.0
        one_hot_data.append(temp)
    return one_hot_data

def neural_network(x_train,y_train,x_test):
    
    print("Defining hyperparameters...")
    epoch = 10
    batch_size = 20
    learning_rate = 0.01
    seed = 100
    
    print("Defining neural network parameters...")
    input_neurons = len(x_train[0])
    hidden_neurons = 100
    output_neurons = number_of_classes
    x = tf.placeholder(tf.float32,[None,input_neurons])
    y = tf.placeholder(tf.float32,[None,output_neurons])
    
    print("Assigining random weights and biases...")
    weights = {'hidden':tf.Variable(tf.random_normal([input_neurons,hidden_neurons],seed=seed)),
               'output':tf.Variable(tf.random_normal([hidden_neurons,output_neurons],seed=seed))}
    biases  = {'hidden':tf.Variable(tf.random_normal([hidden_neurons],seed=seed)),
               'output':tf.Variable(tf.random_normal([output_neurons],seed=seed))}
    
    hidden_layer = tf.add(tf.matmul(x,weights['hidden']),biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    
    output_layer = tf.add(tf.matmul(hidden_layer,weights['output']),biases['output'])
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output_layer))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    predict = tf.argmax(output_layer,1)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        print("training ...")
        for j in range(epoch):
            i=0
            while i < len(x_train):
                start = i
                end = i+batch_size
                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])

                sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})
                i+=batch_size
                
        print("done ! \ntesting ...")
        prediction = sess.run(predict,feed_dict={x:x_test})
        if(prediction[0]==0):
            print('Prediction : Love')
        else:
            print('Prediction : Sad')
    
    
    


# In[51]:


def main(test):
    train = pd.read_csv('my_dataset.csv')
    # print(train.describe())
    x_train,y_train = data_preprocessing(train)
    # print(y_train)
    
    # creating the word corpus using both test and train data
    x_train.extend(test)
    words_corpus = create_corpus(x_train)
    x_train.pop(-1)
    
    # creating count vectors
    x_train = count_vectorize(x_train,words_corpus)
    x_test = count_vectorize(test,words_corpus)
    
    # one hot encoding of the test data
    # for example: [0] -> [1,0] and [1] -> [0,1]
    y_train = one_hot(y_train)

    
    # training and testing 
    neural_network(x_train,y_train,x_test)
    
if __name__ == '__main__':
    
    test_sentence = []
    # temp = raw_input('Enter the test sentence : ')
    temp = "The pain is not on the day of missing our dear ones. The pain is really when you live without them and with their presence in your mind."
    test_sentence.append(temp)
    # calling the main function
    main(test_sentence)

