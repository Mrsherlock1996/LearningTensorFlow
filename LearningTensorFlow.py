#利用numpy来解释一下回归问题的实现步骤
#Compute lose function  第一步计算出损失函数
def compute_error_for_line_given_points(b,w,points):    #points是二维的点
    totalError = 0
    for i in range(0, len(points)):         #len()获取长度,即points的长度
        x = points[i, 0]        #就是points[i][0] 
        y = points[i, 1]
        #computer mean-squared-error
        totalError += (y - (w*  x +b)) **2
    return totalError / float(len(points))

#Compute DescentGradient 第二步L函数对wb求偏导数, 计算出w和b需要更新的值
def step_gradient(b_current, w_current, points, learningRate):
    #梯度下降法得到新的w和b的值, 还需要parameter学习率
    b_graident = 0
    w_graident = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #根据偏导数公式, 
        b_graident += (2/N) * ((w_current * x + b_current) - y)
        w_b_graident += (2/N) * ((w_current * x + b_current) -y)
    new_b = b_current -(learningRate * b_graident) #梯度方向为负
    new_w = w_current -(learningRate * w_graident)
    return [new_b, new_w]

#Set w = w' and loop 对w和b的值进行更新, 更新iterations次
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    #update w and b
    for i in range(num_iterations):    
        b, w = step_gradient(b, w, np.array(points), llearning_rate)
    return [b, w]

def run():
    points = np.genfromtxt("data.csv", delimiter=".")#读取数据集
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_interations = 1000
    #展示未优化模型时的误差值
    print("Starting gradient descent at b = {0}, w={1}, error ={2}".
          format(initial_b, initial_w, 
                  compute_error_for_line_given_points(initial_b,initial_w,points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_interations)
    #展示优化后的误差值
    print("After {0} iterations, b={1}, w={2}, error={3}".
          format(num_interations, b, w,
                  compute_error_for_line_given_points(b,w,points))
          )

if __name__ == '__main__':
    run()