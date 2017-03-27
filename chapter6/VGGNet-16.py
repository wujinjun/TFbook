from datetime import datetime
import math
import time
import tensorflow as tf

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):    #定义一个卷积层函数，输入参数为
                                #（输入tensor名称，本层层名，卷积核的高，卷积核的宽，卷积核的数量，步长的高，步长的宽，网络参数p）  
    n_in = input_op.get_shape()[-1].value    #得到输入的尺寸大小
    
    with tf.name_scope(name) as scope:    #设置scope
        kernel = tf.get_variable(scope+"w",    
            shape=[kh, kw, n_in, n_out],dtype=tf.float32,    #卷积核的大小、通道，以及数据类型
            initializer=tf.contrib.layers.xavier_initializer_conv2d())    #定义初始化方法
        conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1),padding='SAME')    #调用TF的api初始化卷积层
        bias_init_val = tf.constant(0.0,shape=[n_out],dtype=tf.float32)    #定义偏置初始值
        biases = tf.Variable(bias_init_val,trainable=True,name='b')    #定义偏置变量
        z = tf.nn.bias_add(conv,biases)    #将偏置加到卷积结果上
        activation = tf.nn.relu(z,name=scope)    #得到非线性处理后的结果
        p += [kernel, biases]    #把网络参数存入变量p
        return activation

def fc_op(input_op, name, n_out, p):     #定义一个全连接层函数，输入参数为
                            #（输入tensor名称，本层层名，输出通道数，网络参数p）
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:    #设置scope
        kernel = tf.get_variable(scope+"w",
            shape=[n_in,n_out],dtype= tf.float32,        #池化层的输入通道数、输出通道数
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')   #定义偏置初始值
        avtivation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)           #得到非线性处理的结果
        p += [kernel,biases]                                        #把网络存储变量p
        return avtivation

def mpool_op(input_op,name, kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,             #上一层输入
                   ksize=[1,kh,kw,1],      #池化大小
                   strides=[1,dh,dw,1],    #步长大小
                   padding='SAME',name=name)

def inference_op(input_op,keep_prob):
    p = []    #定义一个list用来保存
    conv1_1 = conv_op(input_op,name="conv1_1",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)    #第1.1个卷积层  conv1_1  卷积核3*3  输出通道64  步长1*1
    conv1_2 = conv_op(conv1_1,name="conv1_2",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)    #第1.2个卷积层  conv1_2  卷积核3*3  输出通道64  步长1*1
    pool1 = mpool_op(conv1_2,name="pool1",kh=2,kw=2,dw=2,dh=2)              #第1个池化层，  pool1   池化大小2*2         步长2*2
    
    conv2_1 = conv_op(pool1,name="conv2_1",kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)      #第2.1个卷积层  conv2_1  卷积核3*3  输出通道128  步长1*1
    conv2_2 = conv_op(conv2_1,name="conv2_2",kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)    #第2.2个卷积层  conv2_1  卷积核3*3  输出通道128  步长1*1
    pool2 = mpool_op(conv2_2,name="pool2",kh=2,kw=2,dh=2,dw=2)              #第2个池化层，  pool1   池化大小2*2         步长2*2
    
    conv3_1 = conv_op(pool2,name="conv3_1",kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)      #第3.1个卷积层  conv3_1  卷积核3*3  输出通道256  步长1*1
    conv3_2 = conv_op(conv3_1,name="conv3_2",kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)    #第3.2个卷积层  conv3_2  卷积核3*3  输出通道256  步长1*1
    conv3_3 = conv_op(conv3_2,name="conv3_3",kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)   #第3.3个卷积层  conv3_3  卷积核3*3  输出通道256  步长1*1
    pool3 = mpool_op(conv3_3,name="pool3",kh=2,kw=2,dh=2,dw=2)              #第3个池化层，  pool3   池化大小2*2         步长2*2
    
    conv4_1 = conv_op(pool3,name="conv4_1",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)      #第4.1个卷积层  conv4_1  卷积核3*3  输出通道512  步长1*1
    conv4_2 = conv_op(conv4_1,name="conv4_2",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)    #第4.2个卷积层  conv4_2  卷积核3*3  输出通道512  步长1*1
    conv4_3 = conv_op(conv4_2,name="conv4_3",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)   #第4.3个卷积层  conv4_3  卷积核3*3  输出通道512  步长1*1
    pool4 = mpool_op(conv4_3,name="pool4",kh=2,kw=2,dh=2,dw=2)              #第4个池化层，  pool4   池化大小2*2         步长2*2
    
    conv5_1 = conv_op(pool4,name="conv5_1",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)        #第5.1个卷积层  conv5_1  卷积核3*3  输出通道512  步长1*1
    conv5_2 = conv_op(conv5_1,name="conv5_2",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)      #第5.1个卷积层  conv5_1  卷积核3*3  输出通道512  步长1*1
    conv5_3 = conv_op(conv5_2,name="conv5_3",kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)     #第5.1个卷积层  conv5_1  卷积核3*3  输出通道512  步长1*1
    pool5 = mpool_op(conv5_3,name="pool5",kh=2,kw=2,dh=2,dw=2)                #第5个池化层，  pool5   池化大小2*2         步长2*2
    
    shp = pool5.get_shape()                             #得到pool5层的尺寸
    flattened_shape = shp[1].value * shp[2].value * shp[3].value    #计算三维变量降维成一维以后的长度
    resh1 = tf.reshape(pool5,[-1,flattened_shape],name="resh1")    #得到将三维变量降维成一维变量
    
    fc6 = fc_op(resh1,name="fc6",n_out=4096,p=p)                        #第6个全连接层，输出通道4096
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name="fc6_drop")                #第6个全连接层之后的dropout层
    
    fc7 = fc_op(fc6_drop,name="fc7",n_out=4096,p=p)                     #第7个全连接层，输出通道为4096
    fc7_drop = tf.nn.dropout(fc7,keep_prob,name="fc7_drop")               #第7个全连接层之后的dropout层
    
    fc8 = fc_op(fc7_drop,name="fc8",n_out=1000,p=p)          #第8个全连接层，输出通道为1000
    softmax=tf.nn.softmax(fc8)                       #最后的softmax层
    predictions = tf.argmax(softmax,1)                 #得到softmax的预测值
    return predictions,softmax,fc8,p  

def time_tensprflow_run(session,target,feed,info_string):     #定义计时函数
    num_step_burn_in =10                          #热身10轮迭代
    total_duration = 0.0                         
    total_duration_squared = 0.0
    for i in range(num_batches+num_step_burn_in):
        start_time =time.time()
        _ = session.run(target,feed_dict=feed)         #用sess.run()方法引入feed_dict，方便后面传入keep_prob参数控制dropout层
        duration = time.time()-start_time
        if i >= num_step_burn_in:                 
            if not i%10:    #每十轮迭代打印一次
                print('%s: step %d, duration = %.3f' % (datetime.now(),i-num_step_burn_in,duration))    #打印现在时间，迭代轮数，经历的时间
            total_duration += duration
            total_duration_squared += duration * duration   
    mn = total_duration / num_batches    #计算时间的均值
    vr = total_duration_squared / num_batches - mn * mn    #计算时间的方差
    sd = math.sqrt(vr)    #计算花费时间的标准差
    print('%s: %s across %d step, %.3f +/- %.3f sec/batch' % (datetime.now(),info_string,num_batches,mn,sd))    #打印以上的信息

def run_benchmark():     #定义主函数
    with tf.Graph().as_default():    
        image_size = 224    #定义图片尺寸大小
        images = tf.Variable(tf.random_normal([batch_size,        #随机生产一个大小为224*224的图片
                                 image_size,
                                 image_size,3],
                                 dtype=tf.float32,
                                 stddev=1e-1))
        
        keep_prob = tf.placeholder(tf.float32)        #定义dropout层的keep_prob变量
        predictions,softmax,fc8,p = inference_op(images,keep_prob)      #调用inference_op函数构建VGGNet-16网络
        
        init = tf.global_variables_initializer()        #初始化全局参数
        sess = tf.Session()                     #进入一个会话
        sess.run(init)                        #执行初始化操作
        time_tensprflow_run(sess,predictions,{keep_prob:1.0},"Forward")    #评测forward运算时间,target是求预测值的操作“predictions”
        objective = tf.nn.l2_loss(fc8)        #计算fc8输出的L2-loss
        grad = tf.gradients(objective,p)      #计算梯度的操作
        time_tensprflow_run(sess,grad,{keep_prob:0.5},"Forward-backward")  #target是求梯度的操作"grad"

batch_size = 32    
num_batches = 100
run_benchmark()    #执行主函数