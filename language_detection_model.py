
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.python.ops import rnn


# In[2]:


class lid_model(object):
    def __init__(self,vocab_size=10000, n_classes=50,embed_dim=300, hidden_dim=300,num_layers=3, max_gradient_norm=10., batch_size=32, learning_rate=0.005,
                 learning_rate_decay_factor=0.5, keep_prob=0.75, use_lstm=True, forward_only=False,dtype=tf.float32):
        self.emb_dim=embed_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.cell=tf.contrib.rnn.BasicLSTMCell(embed_dim)
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.keep_prob=keep_prob
        self.num_layers=num_layers
        self.n_classes=n_classes
        self.max_gradient_norm=max_gradient_norm
        self.add_place_holders()
        self.build_graph()
    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)

    def add_place_holders(self):
        self.x=tf.placeholder(tf.int32, shape=[None, None],name="x")
        self.x_mask=tf.placeholder(tf.int32, shape=[None, None],name="x_mask")
        self.y=tf.placeholder(tf.int32, shape=[None,None],name="y")
    def build_graph(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("Embedding") as scope:
            _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.vocab_size, self.embed_dim])
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.x, name="word_embeddings")
            self.word_embeddings = self.dropout(word_embeddings)
        with tf.variable_scope("RNN"):
            output=self.multilayer_bi_rnn(self.cell,self.word_embeddings,self.x_mask,scope)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.hidden_dim, self.n_classes])

            b = tf.get_variable("b", shape=[self.n_classes],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b
        with tf.variable_scope("Loss") as scope:
            self.logits = tf.reshape(pred, [-1, nsteps, self.n_classes])
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                       tf.int32)
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.y)
            self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.labels_pred,self.y),tf.float32))
                #tf.metrics.accuracy(labels=self.y,predictions=self.labels_pred,weights=self.x_mask)
            params = tf.trainable_variables()
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

            gradient = tf.gradients(self.losses, params)
            clipped_gradient, self.norm = tf.clip_by_global_norm(gradient,self.max_gradient_norm )
            self.update = opt.apply_gradients(zip(clipped_gradient, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())
    def bidirectional_rnn_sum(self, cell, inputs, lengths, scope=None):
        name = scope.name or "BiRNN"
        # Forward direction
        with tf.variable_scope(name + "_FW") as fw_scope:
            output_fw, output_state_fw = rnn.dynamic_rnn(cell, inputs, time_major=False, dtype=tf.float32,
                                                       sequence_length=lengths, scope=fw_scope)
        # Backward direction
        inputs_bw = tf.reverse_sequence(inputs, tf.to_int64(lengths), seq_dim=1, batch_dim=0)
        with tf.variable_scope(name + "_BW") as bw_scope:
            output_bw, output_state_bw = rnn.dynamic_rnn(cell, inputs_bw, time_major=False, dtype=tf.float32,
                                                       sequence_length=lengths, scope=bw_scope)

        output_bw = tf.reverse_sequence(output_bw, tf.to_int64(lengths), seq_dim=1, batch_dim=0)

        outputs = output_fw + output_bw
        output_state = output_state_fw + output_state_bw
        return (outputs, output_state)

    def bidirectional_rnn_cat(self, cell, inputs, lengths, scope=None):
        name = scope.name or "BiRNN"
        # Forward direction
        with tf.variable_scope(name + "_FW") as fw_scope:
            output_fw, output_state_fw = rnn.dynamic_rnn(cell, inputs, time_major=False, dtype=tf.float32,
                                                         sequence_length=lengths, scope=fw_scope)
        # Backward direction
        inputs_bw = tf.reverse_sequence(inputs, tf.to_int64(lengths), seq_dim=1, batch_dim=0)
        with tf.variable_scope(name + "_BW") as bw_scope:
            output_bw, output_state_bw = rnn.dynamic_rnn(cell, inputs_bw, time_major=False, dtype=tf.float32,
                                                         sequence_length=lengths, scope=bw_scope)

        output_bw = tf.reverse_sequence(output_bw, tf.to_int64(lengths), seq_dim=1, batch_dim=0)

        outputs = tf.concat((output_fw,output_bw),axis=-1)
        output_state = output_state_fw + output_state_bw
        return (outputs, output_state)

    def multilayer_bi_rnn(self, cell, inputs, mask, scope=None):
        inp = inputs
        srclen = tf.reduce_sum(mask, axis=1)
        for i in range(self.num_layers):
            with tf.variable_scope("Cell%d" % i) as scope:
                out, _ = self.bidirectional_rnn_sum(cell, inp, srclen, scope=scope)
                out=tf.multiply(out,tf.cast(tf.expand_dims(mask,-1),tf.float32))
                inp = self.dropout(out)
        i=self.num_layers-1
        with tf.variable_scope("Cell%d" % i) as scope:
            out, _ = self.bidirectional_rnn_cat(cell, inp, srclen, scope=scope)
            out = tf.multiply(out, tf.cast(tf.expand_dims(mask, -1),tf.float32))
            out = self.dropout(out)
        return out

