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