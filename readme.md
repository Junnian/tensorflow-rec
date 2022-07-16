运行环境：tensorflow 2.2以上版本
# 准备数据
1. feature2type.pickle： 数据到类型映射
2. feature2buckets.pickle: 需要分桶的类别特征对应的词表
3. feature2bin.pickle: 需要分桶的数值特征的对应的边界

# 参考
> https://github.com/wzhe06/SparrowRecSys
```python
import os
import pickle
import pandas as pd
from utils.utils import write_pickle, read_pickle
```


```python
trainingSamples = pd.read_csv("./data/trainingSamples.csv")

feature2type = {}
for feature in trainingSamples.columns:
    if trainingSamples[feature].dtypes == "int64":
        feature2type[feature] = "int32"
    elif trainingSamples[feature].dtypes == "float64":
        feature2type[feature] = "float32"
    else:
        feature2type[feature] = "string"
        
write_pickle(feature2type, "./data/feature2type.pickle")
```


```python
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

feature2buckets = {
    'userGenre1': genre_vocab,
    'userGenre2': genre_vocab,
    'userGenre3': genre_vocab,
    'userGenre4': genre_vocab,
    'userGenre5': genre_vocab,
    'movieGenre1': genre_vocab,
    'movieGenre2': genre_vocab,
    'movieGenre3': genre_vocab
}

write_pickle(feature2buckets, "./data/feature2buckets.pickle")
```

# 配置特征
修改：feature_config.py

# 配置特征的处理方式
修改：config.py

# 训练
运行 train.py


```python

```
