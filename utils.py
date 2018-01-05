
# coding: utf-8

# In[ ]:


import numpy as np
import random


# In[3]:


def read_dataset_x(fname):
    f=open(fname, encoding="utf-8")
    dataset=[]
    vpointer=0
    vocabulary={}
    for line in f:
        line=line.replace("{","")
        line=line.replace("}","")
        line=line.split(":")[1]
        line=line.replace("\"","")
        words=line.split()
        words.insert(0,"BOS")
        words.insert(len(words),"EOS")
        for word in words:
            if word not in vocabulary:
                vocabulary[word]=vpointer
                vpointer+=1
        words=[vocabulary[word] for word in words]
        dataset.append(words)
    return dataset,vocabulary
def read_dataset_y(fname):
    f=open(fname, encoding="utf-8")
    dataset=[]
    vpointer=0
    vocabulary={}
    for line in f:
        line=line.replace("{","")
        line=line.replace("}","")
        line=line.split(":")[1]
        line=line.replace("\"","")
        words=line.split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word]=vpointer
                vpointer+=1
        words=[vocabulary[word] for word in words]
        dataset.append(words)
    return dataset,vocabulary



# In[ ]:


class TextIterator:
    def __init__(self,x_dataset,y_dataset,batch_size=20):
        self.x=x_dataset
        self.y=y_dataset
        self.n=len(x_dataset)
        self.indices=[i for i in range(self.n)]
        random.shuffle(self.indices)
        self.current_index=0
        self.batch_size=batch_size
    def __iter__(self):
        return self
    def reset(self):
        self.current_index=0
    def get_elements_from_index(self,indices):
        lx=[]
        ly=[]
        for i in indices:
            lx.append(self.x[i]) 
            ly.append(self.y[i])
        return lx,ly
    def __next__(self):
        if self.current_index==-1:
            self.reset()
            raise StopIteration
        if (self.current_index+self.batch_size)<self.n:
            indices=self.indices[self.current_index:self.current_index+self.batch_size]
            x,y=self.get_elements_from_index(indices)
            self.current_index+=self.batch_size
            if self.current_index==self.n:
                self.reset()
                raise StopIteration
        else:
            size=self.n-self.current_index
            indices=self.indices[self.current_index:self.current_index+size]
            x,y=self.get_elements_from_index(indices)
            self.current_index=-1
        return x,y


# In[ ]:


#x,_=read_dataset('data/sample.txt')
#y,_=read_dataset('data/sampley.txt')
#t=TextIterator(x,y,4)
#for log,lab in t:
#    print(log,lab)


# In[ ]:


def read_test_dataset(fname,vocabulary):
    f=open(fname, encoding="utf-8")
    dataset=[]
    cnt1=0
    cnt2=0
    for line in f:
        sent=[]
        line=line.replace("{","")
        line=line.replace("}","")
        line=line.split(":")[1]
        line=line.replace("\"","")
        words=line.split()
        for word in words:
            if word in v.keys():
                sent.append(vocabulary[word])
            else:
                sent.append(vocabulary["UNK"])
        dataset.append(sent)
    return dataset
#read_test_dataset("data/test_X_languages_homework.json.txt",v)


# In[ ]:


def add_extras(vocabulary):
    n=len(vocabulary)
    vocabulary["UNK"]=n
    return vocabulary


# In[ ]:




# In[4]:


def prepare_input_feed(model,x,y,max_sen_len):
    n_sen=len(x)
    for i in range(n_sen):
        if len(x[i])>max_sen_len:
            x.pop(i)
            y.pop(i)
    n_sen=len(x)
    seq_len_x=[len(xi) for xi in x]
    max_len_x=max(seq_len_x)
    x_arr=np.zeros((n_sen,max_len_x))
    x_mask=np.zeros((n_sen,max_len_x))
    y_labels = np.zeros((n_sen, max_len_x))
    y_labels = y_labels -1
    for i,xi in enumerate(x):
        label=y[i][0]
        for j,xj in enumerate(xi):
            x_arr[i][j]=xj
            x_mask[i][j]=1
            y_labels[i][j]=label
    return {
        model.x: x_arr,
        model.x_mask: x_mask,
        model.y: y_labels
}

