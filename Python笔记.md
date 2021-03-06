---
title: Python笔记
---

```PYTHON
# 绝对引入
from __future__ import absolute_import
# 绝对除法 / and //
from __future__ import division
# 在Python2中兼容Python3加括号的print
from __future__ import print_function
# 兼容Python3与Python2
from six import string_types, iteritems
```

```Python
# enumerate() 此函数将可遍历的数据对象组合为一个索引序列。
>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

```Python
# numpy.transpose() 交换轴，默认为转置
kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
```

```Python
'''
①tf.flags.DEFINE_xxx() 
②FLAGS = tf.flags.FLAGS 
③FLAGS._parse_flags()
用于帮助我们添加命令行的可选参数。 
也就是说利用该函数我们可以实现在命令行中选择需要设定的参数来运行程序， 
可以不用反复修改源代码中的参数，直接在命令行中进行参数的设定。
'''
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
```


```python
# 参数可视化
tf.summary.histogram(var.op.name + "/activation", var)
tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))
```















