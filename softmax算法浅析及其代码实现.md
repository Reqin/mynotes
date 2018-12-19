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
$$gradient_{\theta_j}=\frac{1}{m}\sum_{i=1}^m(h^i-y^i)\bullet{}x_j^i$$
可以得到参数$\theta$的学习规则：
$$\theta_j:=\theta_j+\alpha(y^i-h^i)\bullet{}x_j^i$$
**现在我们需要使用softmax回归去进行多分类算法模型的学习**
假设我们现在有m类数据，包含n类特征，为了得出每一类的概率，我们就需要有k组权值：
$$\Theta=
\begin{bmatrix}
\theta_1^1&0&0\\
0&1&0\\
0&0&1\\
\end{bmatrix}
$$
k组权值与原始特征经过模型，可以得到在每一类特征下的类概率值：
得到最大的类概率值那一组就是我们的预测值，为了将类概率值转化为概率值，我们需要将其进行归一化处理
可以推出其似然函数为：
$$L(\theta)=\prod_{i=1}^m\prod_{j=1}^k$$

