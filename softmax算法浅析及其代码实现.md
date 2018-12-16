---
title: softmax算法浅析及其代码实现
tags: 新建,模板,小书匠
grammar_cjkRuby: true
---
---

### 算法浅析：
在学习softmax算法之前，你需要阅读[这篇文章](https://blog.csdn.net/qq_36782182/article/details/85009739)了解一些基本概念
对于使用logistics函数来解决二分类的问题来说，我们的损失函数是：
$$loss(\theta)=-\sum_{i=1}^m[y^ilog(h_\theta(x^i)+(1-y^i)log(1-h_\theta(x^i))]$$
对于参数$\theta$进行学习的梯度是：
$$gradient_{\theta_j}=\frac{1}{m}\sum_{i=1}^m(y^i-h^i)x_j^i$$
