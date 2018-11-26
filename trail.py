import math
import os
import tensorflow as tf
import numpy as np
import cPickle
import skimage
import pprint
import tensorflow.python.platform
from keras.preprocessing import sequence
from data_loader import *
import vgg19
import question_generator


from keras.models import Sequential, Model, load_model
from keras.layers import InputLayer, Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, TimeDistributed, Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate, Multiply, Add 
from keras.layers.recurrent import LSTM, SimpleRNN, GRU 
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras.layers import (Dense, Embedding, GRU, Input, LSTM, RepeatVector,
                          TimeDistributed)
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.regularizers import l1_l2

# from keras.preprocessing import image
# from keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import preprocess_input

def read_image(path):
	img = crop_image(path, target_height=224, target_width=224)
	if img.shape[2] == 4:
	 img = img[:,:,:3]

	img = img[None, ...]
	return img

dataset, img_features, train_data = get_data("data_prepro.json","data_img.h5","data_prepro.h5",1);


max_token_len=26;
dim_image=4096;
vocab_size=len(dataset['ix_to_word'].keys())





num_train = train_data['question'].shape[0]
n_words = len(dataset['ix_to_word'].keys())


from keras.layers import Dense, Activation,Input, Add, Masking, Embedding, RepeatVector, BatchNormalization, concatenate, Dropout, TimeDistributed, InputLayer

image_model = Sequential()
image_model.add(Dense(512, input_dim = (4096), activation='elu', name = 'image'))
image_model.add(BatchNormalization())
imageout=image_model.output
imageout=RepeatVector(1)(imageout)

lang_model = Sequential()
lang_model.add(Embedding(vocab_size, 256, input_length=max_token_len, name = 'text'))
lang_model.add(BatchNormalization())
lang_model.add(LSTM(256,return_sequences=True))
lang_model.add(BatchNormalization())
lang_model.add(Dropout(0.3))
lang_model.add(TimeDistributed(Dense(512)))
lang_model.add(BatchNormalization())


intermediate = concatenate([lang_model.output,imageout],axis=1)
intermediate = LSTM(1024,return_sequences=True,dropout=0.5)(intermediate)
intermediate = BatchNormalization()(intermediate)

intermediate = LSTM(1536,return_sequences=True,dropout=0.5)(intermediate)
intermediate = BatchNormalization()(intermediate)
intermediate = (Dropout(0.3))(intermediate)
intermediate = TimeDistributed(Dense(vocab_size,activation='softmax', name='output'))(intermediate)

model=Model(inputs=[image_model.input,lang_model.input],outputs=intermediate)
model.compile('adamax',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit_generator([np.asarray([img_features[i] for i in train_data['img_list'][:200]]),train_data['question'][:200]],epochs=2,steps_per_epoch=100,shuffle = True, verbose = 1)


images = tf.placeholder("float32", [1, 224, 224, 3])

image_val = read_image("bomma.png")

vgg = vgg19.Vgg19()
with tf.name_scope("content_vgg"):
    vgg.build(images)

fc7 = self.sess.run(vgg.relu7, feed_dict={images:image_val})


ixtoword=dataset['ix_to_word']

text_in = np.zeros((1,gen.max_token_len))
text_in[0][:] = np.full((gen.max_token_len,), 0)
text_in[0][0] = 11106

predictions = []
for arg in range(max_token_len-1):
	pred=model.predict([fc7,text_in]);
	tok = np.argmax(pred[0][arg])
	word = gen.id_to_token[tok]
    text_in[0][arg+1] = tok
    predictions.append(word)

print (' '.join(predictions))



