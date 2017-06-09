'''
Matrix factorisation is an embedding model that embeds both user and item in a shared latent space and predicts the rating as an inner product of the embedding
'''

import numpy as np
import tensorflow  as tf
from tensorflow.contrib import keras
from keras import backend as K

tf_graph = tf.get_default_graph()
_sess_config = tf.ConfigProto(allow_soft_placement=True)
_sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=_sess_config, graph=tf_graph)
K.set_session(sess)
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge, Dot
from keras.optimizers import Adam


def build_model(nb_users, nb_items, latent_dim):
    item_input = Input((1,), name='item_input')
    user_input = Input((1,), name='user_input')

    item_x = Embedding(nb_items, latent_dim, name='item_embedding', input_length=1)(item_input)
    item_x = Flatten()(item_x)

    user_x = Embedding(nb_users, latent_dim, name='user_embedding', input_length=1)(user_input)
    user_x = Flatten()(user_x)

    pred = Dot(axes=-1)([user_x, item_x])  # +tf.Variable(0.0)

    model = Model(inputs=[item_input, user_input],
                  outputs=[pred])
    model.compile(loss='mse', optimizer=Adam(), metrics=['mse'])

    return model


if __name__ == '__main__':
    import Data, Utils

    data = Data.Data()
    data.split(ratio=0.1)

    # x = {
    #     'item_input': data.trainset[:, 0],
    #     'user_input': data.trainset[:, 1]
    # }
    # y = data.trainset[:,2]

    x = {
        'item_input': data.raw_data[:, 0],
        'user_input': data.raw_data[:, 1]
    }
    y = data.raw_data[:, 2]

    assert x['item_input'].shape[0] == y.shape[0]
    latent_dim = 10000
    epochs = 10000
    model = build_model(data.nb_users, data.nb_items, latent_dim=latent_dim)
    # model.load_weights('t.h5')
    model.fit(x, y, batch_size=data.trainset.shape[0], validation_split=.9, epochs=epochs, verbose=1, callbacks=[])
    exit(-2)
    model.save('t.h5')

    print(y)
    print(model.predict(x))

    print model.evaluate(x, y, batch_size=data.trainset.shape[0])

    test_x = {
        'item_input': data.testset[:, 0],
        'user_input': data.testset[:, 1]
    }
    test_y = data.testset[:, 2]

    print model.evaluate(test_x, test_y, batch_size=data.testset.shape[0])

    all_data = Data.get_all_test(data.nb_users, data.nb_items)
    pred_x = {
        'item_input': data.trainset[:, 0],
        'user_input': data.trainset[:, 1]
    }
    print model.predict(pred_x)
