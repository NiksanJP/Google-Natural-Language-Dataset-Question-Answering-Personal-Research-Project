import json
import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

def createModel(vocab_size, batchSize):
    #Document
    document_input = tf.keras.layers.Input(batch_shape = [batchSize, None])
    document = tf.keras.layers.Embedding(vocab_size, 256, batch_input_shape = [batchSize, None])(document_input)
    document = tf.keras.layers.GRU(1024, return_sequences = True, stateful = True, recurrent_initializer='glorot_uniform')(document)
    document = tf.keras.layers.Dense(256, activation = 'relu')(document)
    document_model = tf.keras.models.Model(inputs = document_input, outputs = document)
    
    #Question
    question_input = tf.keras.layers.Input(batch_shape = [batchSize, None])
    question = tf.keras.layers.Embedding(vocab_size, 256)(question_input)
    question = tf.keras.layers.GRU(1024, return_sequences = True, stateful = True, recurrent_initializer='glorot_uniform')(question)
    question = tf.keras.layers.Dense(126, activation = 'relu')(question)
    question_model = tf.keras.models.Model(inputs = question_input, outputs = question)
    
    #concat
    model = tf.keras.layers.concatenate([document_model.output, question_model.output])
    model = tf.keras.layers.Dense(vocab_size)(model)
    
    finalModel = tf.keras.models.Model(inputs = [document_input, question_input], outputs = model)
    
    return finalModel

class trainGenSeq_short(Sequence):
    def __init__(self, batchSize):
        self.batchSize = batchSize
        self.trainFiles = os.listdir('D:/Python/Datasets/v1.0/train/')
        self.vocab_Dict, self.vocab_Array = self.loadVocab()
        self.trainingSamples = 307372
    
    def __len__(self):
        return int(self.trainingSamples // self.batchSize)
    
    def getLen(self):
        return int(self.trainingSamples // self.batchSize)
    
    def loadVocab(self):
        myList = []
        for line in open('dict_list.txt'):
            myList.append(line)
        v = set(myList)
        a = {word:index for index, word in enumerate(v)}
        b = np.array(v)
        return a,b
    
    def __getitem__(self, trash):
        print("Here")
        documentStack = np.array()
        titleStack = np.array()
        questionStack = np.array()
        answerStack = np.array()
        
        for file in self.trainFile:
            for line in open('D:/Python/Datasets/v1.0/train/' + file):
                file = json.loads(line)
                
                question = file.get('question_text')
                title = file.get('document_title')
                
                #annotations
                s_Start = file.get('annotations')[0].get('short_answers')[0].get('start_token')
                s_End = file.get('annotations')[0].get('short_answers')[0].get('end_token')
                
                #document
                document = []

                for indexs in file.get('document_tokens'):
                    if indexs.get('html_token') == False:
                        document.append(indexs.get('token'))
                
                #answer
                answer = []
                
                for index in range(s_Start,s_End):
                    if file.get('document_tokens')[index].get('html') == False:
                        answer.append(file.get('document_tokens')[index].get('token'))
                
                #Convert to Array
                question = tf.data.Dataset.fron_tensor_slices(np.array([self.vocab_Dict[i] for i in question]))
                title = tf.data.Dataset.fron_tensor_slices(np.array([self.vocab_Dict[i] for i in title]))
                document = tf.data.Dataset.fron_tensor_slices(np.array([self.vocab_Dict[i] for i in ' '.join(document)]))
                answer = tf.data.Dataset.fron_tensor_slices(np.array([self.vocab_Dict[i] for i in ' '.join(answer)]))
                
                documentStack = np.concatenate(documentStack, document)
                titleStack = np.concatenate(titleStack, title)
                questionStack = np.concatenate(questionStack, question)
                answerStack = np.concatenate(answerStack, answer)
                
                #documentStack.append(document)
                #titleStack.append(title)
                #questionStack.append(question)
                #answerStack.append(answer)
                
                if len(documentStack) == self.batchSize:
                    #documentStack = np.array(documentStack)
                    #questionStack = np.array(questionStack)
                    #answerStack = np.array(answerStack)
                    
                    print("Returned")
                    yield [documentStack, questionStack], answerStack
                    
                    documentStack = []
                    titleStack = []
                    questionStack = []
                    answerStack = []
                
                
batchSize = 64
vocabSize = 20543

model = createModel(vocabSize, batchSize)
model.summary()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')


trainSeq = trainGenSeq_short(batchSize)

history = model.fit(       trainSeq,
                                     epochs = 2,
                                     steps_per_epoch = trainSeq.getLen(),
                                     verbose = True)