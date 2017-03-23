from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)    #导入数据
sess = tf.InteractiveSession()    #开启一个Session

def weight_variable(shape):    #定义一个权重变量
    initial = tf.truncated_normal(shape,stddev=0.1)    #初始化一个正态分布噪声，标准差为0.1
    return tf.Variable(initial)    

def bias_variable(shape):    #定义一个偏置变量
    initial = tf.constant(0.1,shape=shape)    #初始化一个常量，取值为0.1，用来避免死亡节点
    return tf.Variable(initial)    

def conv2d(x,W):    #定义一个卷积层，输入参数为（输入向量X，权重向量W）
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')    #定义一个二维卷积层，

def max_pool_2x2(x):    #定义一个池化层
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')    #调用TF的池化层函数

x = tf.placeholder(tf.float32,[None,784])    #定义输入，条目不限，维度为784
y_ = tf.placeholder(tf.float32,[None,10])    #定义输入，条目不限，类别为10
x_image = tf.reshape(x,[-1,28,28,1])     #对1*785的数据进行reshape，转化为二维数据，-1代表样本不固定，尺寸转化为28*28，通道为1

W_conv1 = weight_variable([5,5,1,32])    #初始化第一个卷积层的权重变量，5*5的核，1个通道，32个不同的卷积核
b_conv1 = bias_variable([32])    #初始化对应于卷积核的偏置
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)    #调用卷积层定义函数，初始化第一个卷积层的激活函数，使用relu激活函数
h_pool1 = max_pool_2x2(h_conv1)    #调用池化层定义函数，初始化第一个卷积层之后的池化层

W_conv2 = weight_variable([5,5,32,64])    #初始化第二个卷积层的权重变量，5*5的核，来自上一层的32个通道，64个不同的卷积核
b_conv2 = bias_variable([64])    #对应于64个核的偏置
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)   
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64,1024])    #初始化第一个全连接层，大小变为7*7*64，共有1024个隐含节点
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)    #定义dropout的placeholder
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)    #定义dropout层，输入是fc1层输出结果以及keep_prob

W_fc2 = weight_variable([1024,10])    
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)    #上一层dropout的输出连接一个softmax层

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))    #求交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)    #使用Adm优化器求得最小的交叉熵

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))    #比较预测结果与真实标签准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))    #求准确率

tf.global_variables_initializer().run()
for i in range(20000):    #进行20000次训练迭代
    batch = mnist.train.next_batch(50)    #使用大小为50的mini-batch
    if i%100==0:    #每迭代100次打印一次
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})     #评测时，dropout比例为1.0
        print("step %d ,training accuracy %g" %(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})    #训练时，dropout比例为0.5

print("test accuracy %g" %accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))    #评测时，dropout比例为1.0