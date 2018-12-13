---
title: 逻辑回归算法的Python实现
tags: 新建,模板,小书匠
grammar_cjkRuby: true
---


### Logistic回归
> Logistic逻辑回归是线性回归模型的一种函数映射,线性回归的预测是根据模型特征的线性叠加，在经过sigmoid函数之后模型就变成了非线性的，在$x=0$的时候梯度很高，在$|x|$越大时梯度越小。Logistic回归被用在二分类问题里面，其定义域为$(0,1)$，在具体问题里面可以看做二分类的某一类的概率。
> * 线性回归模型：
>   $$y=w_0 + w_1x_1+...+w_nx_n$$
>   *其中$x_1,x_2.x_n$是样本的n个特征*
> * sigmod函数：
>   $$f(x)=\frac{1}{1+e^x}$$
>   <center><img src="https://img-blog.csdnimg.cn/20181213135910774.jpg" height="200" width="500" /></center>
>  对于sigmoid函数来说，$x$越大，$y$越趋近于1，$x$越小，$y$越趋近于0，这种变化就是logistic变化。
>  同理，对于各个维度上面的特征来说，经过了线性模型再经过sigmoid函数，其模型为：
>  $$f(x)=\frac{1}{1+e^{-W^TX}}$$
>  对于logistic回归来说，在各个维度的特征经过模型之后，值的绝对值越大就越接近二分类模型的某一类，