---
title: 逻辑回归算法的Python实现
tags: 新建,模板,小书匠
grammar_cjkRuby: true
---


### Logistic回归
#### 原理浅析
Logistic逻辑回归是线性回归模型的一种函数映射,线性回归的预测是根据模型特征的线性叠加，在经过sigmoid函数之后模型就变成了非线性的，在$x=0$的时候梯度很高，在$|x|$越大时梯度越小。Logistic回归被用在二分类问题里面，其定义域为$(0,1)$，在具体问题里面可以看做二分类的某一类的概率。
* 线性回归模型：
  $$f(x)=w_0 + w_1x_1+...+w_nx_n=\Theta^T{X}$$
  *$\Theta=\begin{bmatrix}w_0&w_1&w_2&\cdots&w_n\end{bmatrix}$，$X=\begin{bmatrix}1&x_1&x_2&\cdots&x_n\end{bmatrix}$*
* sigmod函数：
  $$g(x)=\frac{1}{1+e^{-x}}$$
  <center><img src="https://img-blog.csdnimg.cn/20181213135910774.jpg" height="200" width="500" /></center>
对于sigmoid函数来说，其单调可微，而且形似阶跃函数，$x$越大，$y$越趋近于1，$x$越小，$y$越趋近于0，这种变化就是logistic变化。
对于各个维度上面的特征来说，经过了线性模型再经过sigmoid函数，其Logistc模型为：
$$h(x)=g(f(x))=\frac{1}{1+e^{-\Theta^TX}}$$
二分类问题使用sigmoid函数的原因为：
 * 其定义域为概率分布的$(0,1)$
 * 需要在因变量在sigmoid函数上面趋近于0的时候变化梯度，而不是线性变化
 * 在形成损失函数之后形成凸函数

设正例为1，反例为0，在前向估计之后，可以得到：
$$\begin{cases}
P(y=1|x;\theta)=h_\theta(x)\\
P(y=0|x;\theta)=1-h_\theta(x)\\
\end{cases}
$$
可以简化为：
$$P(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}$$
为了做参数估计,对其做最大似然估计：
$$L(\theta)=\prod_{i=1}^{m}(h_\theta(x^i))^{y^i}(1-h_\theta(x^i))^{1-y^i}$$
因为：
* 在求梯度的时候，连乘的直接微分相对复杂，可以使用log把连乘变成相加
* 概率值是浮点数，多个浮点数相乘易造成浮点数下溢，可以使用log把相乘变成相加

所以我们使用对数似然：
$$l(\theta)=log(L(\theta))=\sum_{i=1}^{m}y^ilog(h_\theta(x^i))+(1-y^i)log(1-h_\theta(x^i))$$
参数$\theta$的梯度：
$$\frac{\delta{}l(\theta)}{\delta(\theta_j)}=\sum_{i=1}^m\left(\frac{y^i}{h_\theta(x^i)} - \frac{1-y^i}{1-h_\theta(x^i)}\right)\bullet{}\frac{\delta(h_\theta(x^i))}{\delta\theta_j}=\sum_{i=1}^{m}(y^i-g(\Theta^TX^i))\bullet{}x_j^i$$
将对数似然取负，那么得到的函数就可以当做损失函数：
$$loss=-\sum_{i=1}^{m}y^ilog(h_\theta(x^i))+(1-y^i)log(1-h_\theta(x^i))$$
参数$\theta$的学习规则：
$$\theta_j:=\theta_j+\alpha(y^i-h_\theta(x^i))x_j^i$$
*此时我们做的是随机梯度上升，$\alpha$是超参数，我们定义的学习率*
为了防止过拟合，我们把损失函数加上正则项：
$$loss=-\sum_{i=1}^{m}y^ilog(h_\theta(x^i))+(1-y^i)log(1-h_\theta(x^i))+\lambda\sum_{j=1}^n{\theta_j}^2$$




#### 代码部分
* 前向算法：
  $$h(x)=\frac{1}{1+e^{-\Theta^TX}}$$
  *其中$X$为$x$构成的特征矩阵，$\Theta$是由$w$构成的参数矩阵*
  ```Python?linenums&fancy=0
   # 前向预测算法
	def forward_prediction(x, weights):
		f = np.matmul(x, weights.T)
		return np.divide(1, np.add(1, np.exp(-f)))
  ```
* 损失函数：
  $$J(\theta)=\frac{1}{m}\sum_{i=1}^{m}[(-y^ilog(h_w(x^i))-1-y^i)log(1-h_w(x^i))]+\lambda\sum_{j=1}^n{\theta_j}^2$$
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