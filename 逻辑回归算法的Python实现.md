---
title: 逻辑回归算法的Python实现
tags: 新建,模板,小书匠
grammar_cjkRuby: true
---


### Logistic回归
#### 原理浅析
Logistic逻辑回归是线性回归模型的一种函数映射,线性回归的预测是根据模型特征的线性叠加，在经过sigmoid函数之后模型就变成了非线性的，在$x=0$的时候梯度很高，在$|x|$越大时梯度越小。Logistic回归被用在二分类问题里面，其定义域为$(0,1)$，在具体问题里面可以看做二分类的某一类的概率。
* 线性回归模型：
  $$y=w_0 + w_1x_1+...+w_nx_n$$
  *其中$x_1,x_2.x_n$是样本的n个特征*
* sigmod函数：
  $$f(x)=\frac{1}{1+e^x}$$
  <center><img src="https://img-blog.csdnimg.cn/20181213135910774.jpg" height="200" width="500" /></center>
对于sigmoid函数来说，其单调可微，而且形似阶跃函数，$x$越大，$y$越趋近于1，$x$越小，$y$越趋近于0，这种变化就是logistic变化。
对于各个维度上面的特征来说，经过了线性模型再经过sigmoid函数，其Logistc模型为：
$$h(x)=\frac{1}{1+e^{-W^TX}}$$
二分类问题使用sigmoid函数的原因为：
 * 其定义域为概率分布的$(0,1)$
 * 需要在因变量在sigmoid函数上面趋近于0的时候变化梯度，而不是线性变化
 * 在形成损失函数之后形成凸函数

设正例为1，反例为零，在前向估计之后，可以得到：
$$\begin{cases}
P(y=1|x;\theta)=h_\theta(x)P(y=0|x;\theta)=h_\theta(x)\\
P(y=0|x;\theta)=h_\theta(x)P(y=0|x;\theta)=1-h_\theta(x)\\
\end{cases}
$$

在进行前向估计之后，我们可以得出$y$为输出正例的概率，则输出反例的概率为$1-y$,可以取两者的之比并且取对数得到对数几率,因此logistc回归又称为对数几率回归：
$$log\frac{y}{1-y}=W^TX+b$$
当我们最后的对数几率越大的时候，相对应的结果越好，

#### 代码部分
* 前向算法：
  $$h(x)=\frac{1}{1+e^{-W^TX}}$$
  *其中$X$为$x$构成的特征矩阵，$W$是由$w$构成的参数矩阵*
  ```Python?linenums&fancy=0
   # 前向预测算法
   def forward_prediction(X,W):
       Y_ = X.dot(W.T)
       return np.exp(-Y_)
  ```
* 损失函数：
  $$J(\theta)=\frac{1}{m}\sum_{i=1}^{m}[(-y^ilog(h_w(x^i))-1-y^i)log(1-h_w(x^i))]$$
  矩阵运算表示：
  $$J(\theta)=\frac{1}{m}[-Y^Tlog(H)-((1-Y)^T(1-H))]$$
  为了解决过拟合的问题，给损失函数添加正则项：
  $$J(\theta)=\frac{1}{m}[-Y^Tlog(H)-((1-Y)^T(1-H)) + \frac{\lambda L\theta^2}{2}]$$
  *其中$\lambda$为超参数，自行调节,由于超参数的存在，所以可以不除2*
  $$L：\begin{bmatrix}
  0&0&\cdots&0 \\
  0&1&\cdots&0\\
  \vdots&\vdots&\ddots&\vdots\\
  0&0&\cdots&1\\
  \end{bmatrix}\in R^{(n+1)*(n+1)}$$
  ```Python?linenums&fancy=0
  # 损失函数
  def cost(W,X,Y):
      Y_ = for
      
  ```
* 对于损失函数求导可得到各个参数的梯度：
  $$\frac{\delta}{\delta\theta}J(\theta)=\frac{1}{m}(X^T(h-y)+L\theta)$$
  ```Python?linenums&fancy=0
  #
  ```