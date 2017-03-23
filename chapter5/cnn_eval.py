import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000   #最大训练轮数
batch_size = 128    
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

def variable_with_weight_loss(shape,stddev,wl):    #定义一个有正则化的权重变量类型
    var = tf.Variable(tf.truncated_normal(shape,stddev = stddev))    #定义缺省的权重变量
    if wl is not None:    #如果wl参数非空
        weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')    #对权重进行L2正则，产生一个loss，加到最后总体loss上
        tf.add_to_collection('losses',weight_loss)    #将得到的loss收集到collection中
    return var

cifar10.maybe_download_and_extract()    #下载并解压数据

images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)    #生成训练数据，其中使用16个独立线程，对数据进行Data Augmentation（数据增强），包括水平翻转，随机剪切，随机亮度和对比度，以及对数据的标准化（对数据减去均值、除以方差）

images_test,labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir,batch_size=batch_size)    #生成测试数据

image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])    #产生存储输入的placeholder
label_holder = tf.placeholder(tf.int32, [batch_size])    #产生存储label的placeholder

weight1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=53-2,wl=0.0)    #初始化第一个卷积层的权重，5*5的核，3个通道，64个卷积核
kernel1 = tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME')    #初始化卷积核，步长均为1
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))    #初始化偏置变量
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))    #卷积层conv1，使用relu激活函数
pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')    #池化层pool1，尺寸大小为3*3，步长为2*2
norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)   #在池化层后接一个LRN层进行归一化

weight2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,wl=0.0)    #5*5的核，输入通道为64，共有64个卷积核
kernel2 = tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME')    
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))    #卷积层conv2
norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)   #归一化层norm2
pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')    #池化层pool2

reshape = tf.reshape(pool2,[batch_size,-1])      #全连接层fc1，把每个batch_size都变成一维向量
dim = reshape.get_shape()[1].value    #使用get_shape函数得到reshape数据扁平化后的维数，
weight3 = variable_with_weight_loss(shape=[dim,384],stddev=0.04,wl=0.004)    #初始化权重，输入为dim，隐含节点为384个
bias3 = tf.Variable(tf.constant(0.1,shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape,weight3)+bias3)

weight4 = variable_with_weight_loss(shape=[384,192],stddev=0.04,wl=0.04)    #全连接层fc2
bias4 = tf.Variable(tf.constant(0.1,shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3,weight4)+bias4)

weight5 = variable_with_weight_loss(shape=[192,10],stddev=0.04,wl=0.04)    #softmax层
bias5 = tf.Variable(tf.constant(0.0,shape=[10]))
logits = tf.add(tf.matmul(local4,weight5),bias5)

def loss(logits,labels):     #计算CNN的loss
    labels = tf.cast(labels,tf.int64)    #把label转换成整形
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')    #求交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')    #求交叉熵的均值
    tf.add_to_collection('losses',cross_entropy_mean)    #将交叉熵的loss加入整体的loss中
    return tf.add_n(tf.get_collection('losses'),name='total_loss')    #将整体的loss求和，得到最终的loss

loss = loss(logits,label_holder)    #将logits节点和label_placeholder传入loss函数，输入参数即最后一层的输出logits与标签label
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)    #选择Adam优化器

top_k_op = tf.nn.in_top_k(logits,label_holder,1)    #求输出分数最高的准确率 

sess = tf.InteractiveSession()    #创建一个Session
tf.global_variables_initializer().run()    #初始化全部模型参数

tf.train.start_queue_runners()    #启动图片数据增强的线程

for step in range(max_steps):    #迭代max_steps次
    start_time = time.time()    #记录开始时间
    image_batch,label_batch = sess.run([images_train,labels_train])    #使用sess.run()方法执行images_train,labels_train的计算
    _,loss_value = sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})    #将这个batch的数据传入train_op和loss的计算
    duration = time.time() - start_time     #计算花费的时间
    if step%10 == 0:
        examples_per_sec = batch_size / duration    #计算一秒处理多少个样本
        sec_per_batch = float(duration)    #计算一个样本花费时间
        format_str=('step %d,loss%.2f(%.1f examples/sec; %.3f sec/batch)')    
        print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))    #答应迭代的次数，loss值， 样本/秒， 秒/样本

num_examples = 10000    #测试样本有10000个
import math
num_iter = int(math.ceil(num_examples / batch_size))    #计算多少个batch才能把全部样本评测完
true_count = 0
total_sample_out = num_iter * batch_size    #总共的评测次数
step = 0
while step < num_iter:    #评测的最大迭代次数
    image_batch,label_batch = sess.run([images_test,labels_test])    #使用sess.run（）方法执行images_test,labels_test的计算
    predictions = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})    #将这个batch的数据送入sess.run()计算prediction
    true_count += np.sum(predictions)    #将正确的样本计数+1
    step += 1

precision = true_count / total_sample_out    #计算准确率，准确率=正确样本数/测评样本数
print('precision @ 1 = %.3f' % precision)