import tensorflow as tf
import numpy as np


class Feature_columns:

	# 数值列，可以自定义归一化方法
	def numeric_column(self, column_name, normalizer_fn=None, **kwargs):
		number = tf.feature_column.numeric_column(column_name, normalizer_fn=normalizer_fn)
		return number

	# 数值列 分桶，需要指定分桶边界
	def numeric_bucket_column(self, column_name, **kwargs):
		return tf.feature_column.bucketized_column(
			tf.feature_column.numeric_column(key=column_name, shape=kwargs["shape"] if "shape" in kwargs else (1,),
			                                 default_value=kwargs["default_value"] if "default_value" in kwargs else 0,
			                                 dtype=kwargs["dtype"]),
			boundaries=kwargs["boundaries"])

	# 类表列，按列表顺序编码
	def categorical_column(self, column_name, **kwargs):
		return tf.feature_column.indicator_column(
			tf.feature_column.categorical_column_with_vocabulary_list(key=column_name,
			                                                          vocabulary_list=kwargs["vocabulary_list"],
			                                                          num_oov_buckets=kwargs[
				                                                          "num_oov_buckets"] if "num_oov_buckets" in kwargs else 0,
			                                                          default_value=kwargs[
				                                                          "default_value"] if "num_oov_buckets" in kwargs else -1))

	def multi_categorical_column(self, column_name, num_buckets):
		return tf.feature_column.indicator_column(
			tf.feature_column.categorical_column_with_identity(key=column_name,
			                                                   num_buckets=num_buckets))


if __name__ == "__main__":
	feature_columns = Feature_columns()
	number = feature_columns.numeric_column("number", normalizer_fn=lambda x: (x - 1.0) / 2.0)
	feature_dict = {"number": [1.1, 1.2, 1.3]}
	feature_layer = tf.keras.layers.DenseFeatures(number)
	output = feature_layer(feature_dict)
	print(output)

	cos_index = feature_columns.numeric_bucket_column(column_name='cos', shape=(1,), default_value=0,
	                                                  dtype=tf.dtypes.float32,
	                                                  boundaries=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	product_dict = {"cos": np.random.random(size=(10, 1))}
	feature_layer = tf.keras.layers.DenseFeatures(cos_index)
	output = feature_layer(product_dict)
	print(output)

	# example 1
	age = feature_columns.categorical_column('age',
	                                         vocabulary_list=['50s', '60s', '70s', '80s', '90s'],
	                                         # 值对应的编码位置，"50s"对应[1,0,0,0,0]
	                                         num_oov_buckets=1,
	                                         # 默认为0，直接忽略oov的词。是正数时表示oov的词应该占据的编码位数，num_oov_buckets为1时表示，编码维度=len(vocabulary_list)+1, oov的词应该编码到多的一列
	                                         default_value=-1,
	                                         # 默认为-1, 直接忽略oov的词。是正数时表示oov的词映射的index, 不能和num_oov_buckets同时使用
	                                         )
	feature_dict = {"age": ["91s", '51s', '50s', '60s', '70s', '80s', '90s']}
	feature_layer = tf.keras.layers.DenseFeatures(age)
	output = feature_layer(feature_dict)
	print(output)
	print("=========================")

	# example 2
	# num_oov_buckets为2,oov的词按顺序依次编码为5，6，5，6
	age = feature_columns.categorical_column('age',
	                                         vocabulary_list=['50s', '60s', '70s', '80s', '90s'],
	                                         # 值对应的编码位置，"50s"对应[1,0,0,0,0]
	                                         num_oov_buckets=2,
	                                         # 默认为0，直接忽略oov的词。是正数时表示oov的词应该占据的编码位数，num_oov_buckets为2时表示，编码维度=len(vocabulary_list)+1, oov的词应该编码到多的一列
	                                         default_value=-1,
	                                         # 默认为-1, 直接忽略oov的词。是正数时表示oov的词映射的index, 不能和num_oov_buckets同时使用
	                                         )
	feature_layer = tf.keras.layers.DenseFeatures(age)
	output = feature_layer(feature_dict)
	print(output)
	print("=========================")

	# example 3
	age = feature_columns.categorical_column('age',
	                                         vocabulary_list=['50s', '60s', '70s', '80s', '90s'],
	                                         # 值对应的编码位置，"50s"对应[1,0,0,0,0]
	                                         num_oov_buckets=0,
	                                         # 默认为0，直接忽略oov的词。是正数时表示oov的词应该占据的编码位数，num_oov_buckets为2时表示，编码维度=len(vocabulary_list)+1, oov的词应该编码到多的一列
	                                         default_value=0,
	                                         # 默认为-1, 直接忽略oov的词。是正数时表示oov的词映射的index, 不能和num_oov_buckets同时使用
	                                         )
	feature_layer = tf.keras.layers.DenseFeatures(age)
	output = feature_layer(feature_dict)
	print(output)
	print("=========================")

	multi_category = feature_columns.multi_categorical_column("multi", num_buckets=50)

	feature_dict = {"multi": [[4, 5, 7, 10], [1, 2, 9, 30], [0, 1, 2, 3]]}  # 必须是大于等于0的数，不能是负数
	feature_layer = tf.keras.layers.DenseFeatures(multi_category)
	output = feature_layer(feature_dict)
	print(output)
	print("=========================")
