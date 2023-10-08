# 底层实现
#线性回归的实现步骤：
# 1.定义loss函数
# 2.定义梯度下降
# 3.循环求解


#定义LOSS函数  这里使用
def compute_error_for_line_given_points(b,w,points):
    total_errors = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        total_errors += (y-(w*x+b))**2