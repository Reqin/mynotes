---
title: 逻辑回归算法的Python实现
tags: 新建,模板,小书匠
grammar_cjkRuby: true
---


### Logistic回归
> Logistic逻辑回归是线性回归模型的一种函数映射,线性回归的预测是根据模型特征的线性叠加，在经过sigmoid函数之后模型就变成了非线性的，在$x=0$的时候梯度很高，在$|x|$越大时梯度越小。
> * 线性回归模型：
>   $$y=w_0 + w_1x_1+...+w_nx_n$$
>   *其中$x_1,x_2.x_n$是样本的n个特征*
> * sigmod函数：
>   $$f(x)=\frac{1}{1+e^x}$$
>   <img src="https://img-blog.csdnimg.cn/20181213135910774.jpg" />