import tensorflow as tf
import data_utils as data_utils
import math
from Beam_Attention_Decoder import BeamAttentionDecoder
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops

class Seq2SeqBiDirectionalModel(object):

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 embed_dim,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 keep_prob,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32,
                 beam_size=10,
                 src_embeddings=None,
                 target_embeddings=None,):
        self.emb_dim=embed_dim
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        # Create the internal multi-layer cell for our RNN.
        def single_cell(dim):
            return tf.contrib.rnn.GRUCell(dim)

        if use_lstm:
            def single_cell(dim):
                return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(dim),input_keep_prob=keep_prob)
        encoder_cell = single_cell(self.emb_dim)
        self.encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, input_keep_prob=keep_prob)
        if num_layers > 1:
            self.encoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell(self.emb_dim) for _ in range(num_layers)])
        decoder_cell = single_cell(self.emb_dim)
        self.decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=keep_prob)
        if num_layers > 1:
            self.decoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell(self.emb_dim) for _ in range(num_layers)])
        self.decoder_hidden_units = self.decoder_cell.output_size
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )
        self.decoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_inputs'
        )
        self.decoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_inputs_length',
        )
        self.targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='targets'
        )
        self.loss_weights = tf.placeholder(
            shape=(None, None),
            dtype=tf.float32,
            name='loss_weights'
        )
        #self.target_seq_len=tf.placeholder(dtype=tf.int32)

        with tf.variable_scope("Embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
            self.src_embedding_matrix = tf.get_variable(
                name="src_embedding_matrix",
                shape=[self.source_vocab_size, self.emb_dim],
                initializer=tf.constant_initializer(src_embeddings),
                dtype=tf.float32,
                trainable=True)

            self.target_embedding_matrix = tf.get_variable(
                name="target_embedding_matrix",
                shape=[self.target_vocab_size, self.emb_dim],
                initializer=tf.constant_initializer(target_embeddings),
                dtype=tf.float32,
                trainable=True)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.src_embedding_matrix ,self.encoder_inputs)
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.target_embedding_matrix,self.decoder_inputs)
            #print("embedding matrix dims: ",self.embedding_matrix.get_shape())

        """
        Both encoder and decoder have time-major inputs and outputs. The dimensions of the input, and output are 
        [seq_length,batch_size,emb_dim]
        """

        with tf.variable_scope("BidirectionalEncoder") as scope:
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_states,
              encoder_bw_states)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32)
            )

            self.encoder_fw_outputs=encoder_fw_outputs

            '''Can use either of addition or concatenation reduction operation here
            For concatenation, the size of decoder cell is increased by 2. We have used addition here, because it takes up
            less space, and there is no established difference between the two.
            '''
            self.encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)

            #Required for addition of multi-layer rnn cells, tuple structure for layer, and c,h gate in lstm
            self.encoder_state=[]
            #print("enc state len: ",len(encoder_fw_state))
            for encoder_fw_state,encoder_bw_state in list(zip(encoder_fw_states,encoder_bw_states)):
                if isinstance(encoder_fw_state, tf.contrib.rnn.LSTMStateTuple):
                    encoder_state_c = tf.add(
                        encoder_fw_state.c, encoder_bw_state.c, name='bidirectional_concat_c')
                    encoder_state_h = tf.add(
                        encoder_fw_state.h, encoder_bw_state.h, name='bidirectional_concat_h')
                    self.encoder_state.append(tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h))
                elif isinstance(encoder_fw_state, tf.Tensor):
                    self.encoder_state.append(tf.add(encoder_fw_state, encoder_bw_state, name='bidirectional_concat'))
            self.encoder_state=tuple(self.encoder_state)

        self.outputs,self.infer\
            =BeamAttentionDecoder(decoder_cell=self.decoder_cell,decoder_inputs_embedded=self.decoder_inputs_embedded,
                                          encoder_outputs=self.encoder_outputs,encoder_state=self.encoder_state,
                                          target_embedding_matrix=self.target_embedding_matrix,emb_dim=self.emb_dim,
                                          target_vocab_size=self.target_vocab_size,beam_size=beam_size,dtype=dtype,
                                          max_possible_length=tf.reduce_max(self.encoder_inputs_length))

        with tf.variable_scope("Loss") as scope:
            #logits=tf.nn.embedding_lookup(self.target_embedding_matrix,self.outputs)
            logits = tf.transpose(self.outputs, [1, 0, 2])
            targets = tf.transpose(self.targets, [1, 0])
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets,
                                                         weights=self.loss_weights)
        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

            gradient = tf.gradients(self.loss, params)
            clipped_gradient, self.norm = tf.clip_by_global_norm(gradient,max_gradient_norm)
            self.update=opt.apply_gradients(zip(clipped_gradient, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())




