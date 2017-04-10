
# coding: utf-8

# In[1]:

cd ../chapter5/models-master/tutorials/rnn/ptb


# In[2]:


import time
import numpy as np
import tensorflow as tf
import reader


# In[3]:

class PTBInput(object):
    
    def __init__(self,config,data,name=None):
        self.batch_size = batch_size = config.batch_size    #从config中读取参数存到本地变量
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = (len(data) // batch_size - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data,batch_size,num_steps,name=name)


# In[4]:

class PTBModel(object):
    def __init__(self,is_training,config,input_):
        self._input = input_
        
        batch_size = input_.batch_size    #从input_中读取参数存到本地变量
        num_steps = input_.num_steps
        size = config.hidden_size    #从config中读取参数存到本地变量，隐含节点个数
        vocab_size = config.vocab_size
        def lstm_cell():    #使用tf.contrib.rnn.BasicLSTMCell函数设置默认的LSTM单元
            return tf.contrib.rnn.BasicLSTMCell(size,forget_bias=0.0,state_is_tuple=True)    #size是隐含节点个数，forgets_bias是forget gate的bias，
        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:    #如果是在训练状态且keepprob<1则在前面的lstm_cell之后接一个Dropout层
            def attn_cell():    #使用tf.contrib.rnn.DropoutWrapper函数设置一个dropout层
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(),output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)],state_is_tuple=True)#用tf.contrib.rnn.MultiRNNCell函数堆叠前面构造的lstm_cell

        self._initial_state = cell.zero_state(batch_size, tf.float32)    #设置LSTM单元的初始化状态为0    

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(      #embedding是一个向量， 将one-hot编码格式的单词转化为向量表达形式
                "embedding", [vocab_size,size],dtype=tf.float32)    #vocab_size是词汇表数，每个单词向量表达所需的维数为size   分别构成embedding的行和列
            inputs = tf.nn.embedding_lookup(embedding,input_.input_data)    #查询单词对应的向量表达获得inputs
        if is_training and config.keep_prob <1:    #如果是训练状态，还要在后面加上一层dropout层
            inputs = tf.nn.dropout(inputs,config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):     #将下面的操作设为RNN
            for time_step in range(num_steps):    #设置步数，用来限制梯度在反向传播过程步数
                if time_step > 0: tf.get_variable_scope().reuse_variables()    #第二次循环开始设置复用变量
                (cell_output,state) = cell(inputs[:,time_step,:],state) #inputs的三个维度，第1个代表batch中的第几个样本，第2个代表样本中的第几个单词，第三个代表单词的
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs,1),[-1,size]) 
        softmax_w = tf.get_variable(
                    "softmax_w",[size,vocab_size],dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b",[vocab_size],dtype=tf.float32)
        logits = tf.matmul(output,softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(   #用这个函数来计算targets和logits的偏差
                [logits],
                [tf.reshape(input_.targets,[-1])],
                [tf.ones([batch_size * num_steps],dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size    #汇总batch的误差，在计算每个样本的误差
        self._final_state = state    #保留最终的状态

        if not is_training:
            return
        self._lr = tf.Variable(0.0,trainable=False)   #定义学习率。且设置为不可训练
        tvars = tf.trainable_variables()    
        grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),  #计算tvars的梯度，设置梯度的最大范数,
                                        config.max_grad_norm)    #这个Gradient Cliping方法，控制梯度的最大范数，某种程度上有正则化的效果，防止梯度爆炸问题
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads,tvars),    #定义一个训练操作，将clip过的梯度应用到所有了训练的参数上
                        global_step=tf.contrib.framework.get_or_create_global_step())    #生成全局统一的训练步数

        self._new_lr = tf.placeholder(
                    tf.float32,shape=[],name="new_learning_rate")   #控制学习速率
        self._lr_update = tf.assign(self._lr,self._new_lr)  #将新的学习速率赋值给当前的学习速率_lr
        
    def assign_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self._new_lr:lr_value})
    
    @property
    def input(self):
        return self._input
    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def cost(self):
        return self._cost
    
    @property
    def final_state(self):
        return self._final_state
    
    @property
    def lr(self):
        return self._lr
    
    @property
    def train_op(self):
        return self._train_op


# In[5]:

class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4 
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


# In[6]:

class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


# In[7]:

class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


# In[8]:

class TestConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


# In[9]:

def run_epoch(session,model,eval_op=None,verbose=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)    #初始化获得初始状态
    
    fetches = {
        "cost":model.cost,
        "final_state":model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op    #创建结果的字典表
    
    for step in range(model.input.epoch_size):    #训练epoch_size
        feed_dict = {}
        for i,(c,h) in enumerate(model.initial_state):    #每次把state装入feed_dict
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches,feed_dict)   #跑起
        cost = vals["cost"]     #得到cost
        state = vals["final_state"]   #得到state
        
        costs += costs    #累加cost
        iters += model.input.num_steps   #累加迭代次数，
        
        if verbose and step % (model.input.epoch_size // 10) == 10:    #每隔10次做一次展示
            print("%.3f perplexity:%.3f speed: %.0f wps" %
                 (step * 1.0 / model.input.epoch_size,np.exp(costs/iters),
                 iters * model.input.batch_size / (time.time()-start_time)))
                  
    return np.exp(costs / iters)
            


# In[10]:

data_path='/home/wjj/TFbook/chapter7/simple-examples/data/'
raw_data = reader.ptb_raw_data(data_path)     #直接读取解压后的数据
train_data,valid_data,test_data,_ = raw_data    #将解压后的数据分别存为训练数据和验证数据以及测试数据
config = SmallConfig()    #使用SmallConfig的配置
eval_config = SmallConfig()  #测试配置eval_config需和训练配置一致
eval_config.batch_size = 1
eval_config.num_steps = 1


# In[12]:




# In[ ]:

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)     #
    
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config,data=train_data,name="TrainInput")
        with tf.variable_scope("Model",reuse = None, initializer=initializer):
            m = PTBModel(is_training=True,config=config,input_=train_input)
    
    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config,data=valid_data,name="ValidInput")
        with tf.variable_scope("Model",reuse = True, initializer=initializer):
            mvalid = PTBModel(is_training=False,config=config,input_=valid_input)
            
    with tf.name_scope("Test"):
        test_input = PTBInput(config=config,data=test_data,name="TestInput")
        with tf.variable_scope("Model",reuse = True, initializer=initializer):
            mtest = PTBModel(is_training=False,config=config,input_=test_input)
    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i+1-config.max_epoch,0.0)
            m.assign_lr(session,config.learning_rate * lr_decay)
            
            print("Epoch: %d Learning rate: %.3f" % (i+1,session.run(m.lr)))
            
            train_perplexity = run_epoch(session,m,eval_op=m.train_op,verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i+1,train_perplexity))
            
            valid_perplexity = run_epoch(session,mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i+1,valid_perplexity))
            
        test_perplexity = run_epoch(session,mtest)
        print("Test Perplexity: %.3f" % test_perplexity)
            


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



