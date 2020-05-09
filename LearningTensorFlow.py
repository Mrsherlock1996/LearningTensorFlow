#手写基于tensor而不是API的代码
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os
#设计一个环境变量, 无关的信息是CPP打印出来的, 只需要给他赋值即可 0就是全部打印, 2就是只打印error信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x: [60k, 28, 28],  60k个图片, 每个图片28x28
# y: [60k]    每个图片都带有一个label
(x, y), _ = datasets.mnist.load_data()   #自动下载mnist数据集 python中( )表示元组,即不可改变的类型
# x: [0~255] => [0~1.]  方便计算
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  #数据转换成tensor类对象
#y: [0-9]
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))


train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)  #不想一下把60000个数据都拿出来,一次拿128个来训练
'''
tf.data.Dataset.from_tensor_slices()
语义解释：from_tensor_slices，从张量的切片读取数据。
工作原理：将输入的张量的第一个维度看做样本的个数，沿其第一个维度将tensor切片，得到的每个切片是一个样本数据。实现了输入张量的自动切片。
输入数据格式/要求：
        1）可以是numpy格式，也可以是tensorflow的tensor的格式，函数会自动将numpy格式转为tensorflow的tensor格式
        2）输入可以是一个tensor
                                   或   一个tensor字典（字典的每个key对应的value是一个tensor，要求各tensor的第一个维度相等）
                                   或    一个tensor  tuple（tuple 的每个元素是一个tensor，要求各tensor的第一个维度相等）。
.batch()方法: 一次喂入神经网络的数据量（batch size）
'''
train_iter = iter(train_db)  #迭代器,返回迭代器本身
sample = next(train_iter)    #开始访问迭代器, 返回迭代器的下一个元素
print('batch:', sample[0].shape, sample[1].shape)


# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# [dim_in, dim_out], [dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3  #learning rate 0.0001

for epoch in range(10): # 重复对60000个数据训练10次
    for step, (x, y) in enumerate(train_db):
        ''' 
        60000个数据, 一次拿128个, 需要拿468次,即对[128,28,28,3]的x训练468次
        相当于分了468个batch,每个batch的x是如下:
             x:[128, 28, 28]
             y: [128]

        [128, 28, 28] => [128, 28*28]
        '''
        x = tf.reshape(x, [-1, 28 * 28])
        with tf.GradientTape() as tape:
            #创建一个梯度带, 到时候可以一步得出很多梯度值
            # 这个函数是针对tf.Variable对象的
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b, 256] + [b, 256]
            h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b, 128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            out = h2@w3 + b3

            # compute loss
            # out: [b, 10]
            # y: [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y_onehot - out)
            # mean: scalar
            loss = tf.reduce_mean(loss)
            '''
            tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
                reduce_mean(input_tensor,
                                axis=None,
                                keep_dims=False,
                                name=None,
                                reduction_indices=None)
                第一个参数input_tensor： 输入的待降维的tensor;
                第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
                第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
                第四个参数name： 操作的名称;
                第五个参数 reduction_indices：在以前版本中用来指定轴，已弃用;
            '''

        # compute gradients一步计算超多梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print(grads)
        # w1 = w1 - lr * w1_grad      #w1参与运算可能会修改w1的类型
        w1.assign_sub(lr * grads[0])  #原地更新, 不会因为w1做运算了而改变w1的类型
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        '''
        tf.assign(ref, value, validate_shape=True, use_locking=None, name=None)
            释义：将 value 赋值给 ref，即 ref = value            
            ref，变量
            value，值
            validate_shape，默认 True，值 shape 需要与变量 shape 一致；若设为 False，则值 shape 覆盖变量 shape
            use_locking，默认 False，若为 True，则受锁保护
            name，名称
            
        tf.assign_add(ref, value, use_locking=None, name=None)
            释义：将值 value 加到变量 ref 上， 即 ref = ref + value            
            ref，变量
            value，值
            use_locking，默认 False, 若为 True，则受锁保护
            name，名称
            
        tf.assign_sub(ref, value, use_locking=None, name=None)
            释义：变量 ref 减去 value值，即 ref = ref - value        
            ref，变量
            value，值
            use_locking，默认 False, 若为 True，则受锁保护
            name，名称
        '''

        #if step % 100 == 0:
            #print(epoch, step, 'loss:', float(loss))
        print(epoch, step, 'loss:', float(loss))