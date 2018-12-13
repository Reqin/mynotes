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
对于sigmoid函数来说，$x$越大，$y$越趋近于1，$x$越小，$y$越趋近于0，这种变化就是logistic变化。
对于各个维度上面的特征来说，经过了线性模型再经过sigmoid函数，其Logistc模型为：
$$f(x)=\frac{1}{1+e^{-W^TX}}$$
二分类问题选择logistc模型的原因为：
 * 其定义域为概率分布的$(0,1)$
 * 需要在因变量在sigmoid函数上面趋近于0的时候变化梯度，而不是线性变化
 * 在形成损失函数之后形成凸函数
 
 #### 代码部分
 * 前向算法
 $$f(x)=\frac{1}{1+e^{-W^TX}}$$
