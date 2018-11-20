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
>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```