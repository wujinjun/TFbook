from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activatitions(t):    #打印每层结构的函数
    print(t.op.name,' ',t.get_shape().as_list())    #打印“层的名字”，“tensor的尺寸”

def inference(images):
    parameters = []    #AlexNet中所有需要训练的模型，存入parameter
    with tf.name_scope('conv1') as scope:    #相当于局部namespace,用以区分不同层的相同组件，conv/XXX
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64],dtype = tf.float32,stddev=1e-1),name='weights')    #11*11的卷积核，初始化为标准差为0.1的正太分布变量
        conv = tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')    #步长为4*4，通道为1
        biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')    #初始化偏置为0.0，尺寸为64的常量
        bias = tf.nn.bias_add(conv,biases)    #将偏置加到conv上
        conv1 = tf.nn.relu(bias,name=scope)    #使用relu函数对结果进行非线性处理
        print_activatitions(conv1)    #打印conv1层
        parameters += [kernel,biases]    #将这一层的可训练参数kernel和biases添加到parameters
    lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')    #对前面输出的tensor进行LRN处理，所有参数使用AlexNet论文中推荐的参数
    pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')    #最大池化处理，池化尺寸为3*3，步长为2*2
    print_activatitions(pool1)    #打印pool1层
    
    with tf.name_scope('conv2') as scope:    #conv2,lrn2,pool2
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
        print_activatitions(conv2)
    lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
    print_activatitions(pool2)
    
    with tf.name_scope('conv3') as scope:    #conv3,
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
        print_activatitions(conv3)
        
    with tf.name_scope('conv4') as scope:    #conv4
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
        print_activatitions(conv4)
    
    with tf.name_scope('conv5') as scope:    #conv5
        kernel = tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
        print_activatitions(conv5)
    pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
    print_activatitions(pool5)
    
    return pool5, parameters

def time_tensorflow_run(session,target,info_string):
    num_step_burn_in = 10    #10轮热身迭代,因为显存加载，cache命中等问题跳过前10轮迭代
    total_duration = 0.0    #用来计算总时间
    total_duration_squared = 0.0    #计算平方和计算方差
    for i in range(num_batches + num_step_burn_in): 
        start_time = time.time()    #记录开始时间
        _ = session.run(target)    #每轮迭代由此条语句执行
        duration = time.time() - start_time    #计算花费的时间
        if i >= num_step_burn_in:    #迭代次数超过热身阶段后
            if not i%10:    #每隔10次迭代打印一次
                print('%s: step %d, duration = %.3f' % (datetime.now(),i - num_step_burn_in, duration))
            total_duration += duration    #加上时间
            total_duration_squared += duration * duration    #求花费时间的平方和
    mn = total_duration / num_batches    #求平均耗时
    vr = total_duration_squared / num_batches - mn * mn    #求方差
    sd = math.sqrt(vr)    #求标准差
    print('%s: %s across %d steps,%.3f +/- %.3f sec / batch' % (datetime.now(),info_string,num_batches,mn,sd))    #打印以上信息

def run_benchmark():
    with tf.Graph().as_default():    #定义默认的Graph方便后面使用
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))    #使用随机的数据，用来计算Forward和Backward耗时情况
        pool5,parameters = inference(images)    #用inference函数构建网络
        init = tf.global_variables_initializer()    #初始化所有参数
        sess = tf.Session()    #创建一个会话
        sess.run(init)    
        time_tensorflow_run(sess,pool5,"Forward")    #统计前向运算时间
        
        objective = tf.nn.l2_loss(pool5)    #使用L2_loss计算pool5的loss
        grad = tf.gradients(objective,parameters)    #求网络梯度的操作
        time_tensorflow_run(sess,grad,"Forward-backward")    #统计backward的运算时间

run_benchmark()    #执行主函数
