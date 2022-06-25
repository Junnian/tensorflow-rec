from feature_config import numerical_features, categorical_features
import tensorflow as tf
from utils.utils import read_pickle

# 设置显存使用按需增长
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

# 载入准备的文件
feature2buckets = read_pickle("./data/feature2buckets.pickle")
# feature2bin = read_pickle("/home/support/cjn/project/recommend_research/auto_homepage_feeds/model/tensorflow-FM//utils/feature2bin-7.pickle")
feature2dtype = read_pickle("./data/feature2type.pickle")

# 配置feature_columns
# 具体配置的参数可以看 model_utils.feature_columns的方法的参数

categorical_column_list = [
	{
		"column_name": feature,
		"dtype": feature2dtype[feature],  # 输入数据里面是什么类型这里就是什么类型, 一般是string\float32\int
		"feature_type": "categorical_column",
		"vocabulary_list": feature2buckets[feature]
	} for feature in categorical_features
]

numeric_column_list = [
	{
		"column_name": feature,
		"feature_type": "numeric_column"
	} for feature in numerical_features
]

data_config = {
	"batch_size": 128,
	"label_name": "rating",
	"shuffle": 1,
	"categorical_column_list": categorical_column_list,
	"numeric_bucket_column_list": numeric_column_list,
	"feature2buckets": feature2buckets,
	"feature2dtype": feature2dtype,
	"train_val_data_source": "./data/trainingSamples.csv",
	"test_data_source": "./data/testSamples.csv"
}

# 配置模型相关配置增加
model_config = {
	"latent_feature_dim": 10,
	"learn_rate": 0.001,
	"epochs": 100,
	"patience": 10,
	"model_id": "demo",  # model_name + feature
	"model_save_path": "./ckpt/demo"
}
