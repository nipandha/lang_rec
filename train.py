
# coding: utf-8

# In[3]:


import tensorflow as tf
import argparse
from utils import read_dataset_y,read_dataset_x,read_test_dataset,TextIterator,add_extras,prepare_input_feed
from language_detection_model import lid_model
import time
import os
import numpy as np
# In[ ]:


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--log_dir', type=str, default='logs', metavar='logdir',
                    help="logs directory")
parser.add_argument('--emb_dim', type=int, default=300, metavar='emb_dim',
                    help="Embedding vector dimension")
parser.add_argument('--num_layers', type=int, default=2, metavar='numlayers',
                    help="Depth of RNN")
parser.add_argument('--num_epochs', type=int, default=50, metavar='numepochs',
                    help="Depth of RNN")
parser.add_argument('--batch_size', type=int, default=2, metavar='batchsize',
                    help="batch size")
parser.add_argument('--max_seq_len', type=int, default=100, metavar='maxseqlen',
                    help="Maximum allowed sentence length")

parser.add_argument('--learning_rate', type=float, default=0.0004, metavar='lr',
                    help="learning rate")
parser.add_argument('--learning_rate_decay_factor', type=float, default=0.5, metavar='logdir',
                    help="learning rate decay factor")
parser.add_argument('--keep_prob', type=float, default=0.75, metavar='keepprob',
                    help="probability of a number not being dropout")
parser.add_argument('--checkpoint_dir', type=str, default='ckpts', metavar='chkptdir',
                    help="checkpoint directory")
parser.add_argument('--x_train_path', type=str, default='', metavar='trainX',
                    help="training data path")
parser.add_argument('--y_train_path', type=str, default='', metavar='trainY',
                    help="training labels path")
args = parser.parse_args()
timestr = time.strftime("%Y%m%d%H%M%S")
log_file=args.log_dir+"/log"+timestr+".txt"
dtype = tf.float16 # else tf.float32



max_gradient_norm=10.
min_learning_rate=0.00001




checkpoint_path = os.path.join(args.checkpoint_dir, "train.ckpt")


# In[ ]:


def create_model(session):
    #Use forward only for decode
    """Create translation model and initialize or load parameters in session."""
    model = lid_model(batch_size=args.batch_size,num_layers=2)
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")

    return model


# In[ ]:


def train():
    flog = open(log_file, 'a')
    x_dataset,v=read_dataset_x(args.x_train_path)
    y_dataset,classes=read_dataset_y(args.y_train_path)
    v=add_extras(v)
    n=len(x_dataset)
    n1=int(0.8*n)
    train_iterator=TextIterator(x_dataset[0:n1],y_dataset[0:n1],args.batch_size)
    val_iterator=TextIterator(x_dataset[n1:],y_dataset[n1:],args.batch_size)



    with tf.Session() as sess:
    # Create model.

        model = create_model(sess)

        # This is the training loop.
        valid_acc=[]
        max_acc=0
        for epoch in range(args.num_epochs):
            

            i=0
            print(i)
            i+=1
            for batch_X,batch_Y in train_iterator:
                
                start_time = time.time()
                #Prepare data inplaceholder format
                input_feed=prepare_input_feed(model, batch_X, batch_Y,args.max_seq_len)
                output_feed = [model.accuracy]
                op = sess.run(output_feed, input_feed)
                #print(np.array(op).shape)
                print(op)
                output_feed = [model.update,  # Update Op that does SGD.
                              model.norm,  # Gradient norm.
                              model.losses]  # Loss for this batch.
                _,_,loss = sess.run(output_feed, input_feed)
                #print("Loss at batch ",i,loss)

            acc=0
            n=0
            for batch_X,batch_Y in val_iterator:
                input_feed=prepare_input_feed(model, batch_X, batch_Y,args.max_seq_len)
                output_feed=[model.accuracy]
                v=sess.run(output_feed, input_feed)
                acc+=v[0]
                n+=1
            acc/=n
            print("Val acc at epoch ",epoch,acc)
            #If average epoch loss stays constant for 3 epochs decrease learning rate
            if len(valid_acc) > 2 and (np.min(valid_acc[-3])>acc):
                sess.run(model.learning_rate_decay_op)
                lr=sess.run(model.learning_rate)
                if lr<min_learning_rate:
                    print("Reached convergence at epoch " + str(epoch) + "\n")
                    break
            valid_acc.append(acc)

            #If average loss over the epoch has reduced, save checkpoint
            if(acc > max_acc ):
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                print("Saved checkpoint")
               

            #Store graph
            #writer = tf.summary.FileWriter(graph_path, sess.graph)
            #writer.close()'''
    
train()
