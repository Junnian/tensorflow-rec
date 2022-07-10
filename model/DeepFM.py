import tensorflow as tf
from model.BasicModel import BasicModel


class ReduceLayer(tf.keras.layers.Layer):
	def __init__(self, axis, op='sum', **kwargs):
		super().__init__()
		self.axis = axis
		self.op = op
		assert self.op in ['sum', 'mean']

	def build(self, input_shape):
		pass

	def call(self, input, **kwargs):
		if self.op == 'sum':
			return tf.reduce_sum(input, axis=self.axis)
		elif self.op == 'mean':
			return tf.reduce_mean(input, axis=self.axis)
		return tf.reduce_sum(input, axis=self.axis)


class DeepFM(BasicModel):
	def __init__(self, inputs, deep_columns, cat_columns, embedding_columns):
		self.inputs = inputs
		self.deep_columns = deep_columns
		self.cat_columns = cat_columns
		self.embedding_columns = embedding_columns
		self.model = self.build_model()

	def build_model(self):
		# build firet_order_feature
		first_order_cat_feature = tf.keras.layers.DenseFeatures(self.cat_columns)(self.inputs)
		first_order_cat_feature = tf.keras.layers.Dense(1, activation=None)(first_order_cat_feature)
		first_order_deep_feature = tf.keras.layers.DenseFeatures(self.deep_columns)(self.inputs)
		first_order_deep_feature = tf.keras.layers.Dense(1, activation=None)(first_order_deep_feature)

		first_order_feature = tf.keras.layers.Add()([first_order_cat_feature, first_order_deep_feature])

		# build second order feature
		second_order_cat_columns = []
		for feature_emb in self.embedding_columns:
			feature = tf.keras.layers.Dense(64, activation=None)(feature_emb)
			feature = tf.keras.layers.Reshape((-1, 64))(feature)
			second_order_cat_columns.append(feature)
		second_order_deep_columns = tf.keras.layers.DenseFeatures(self.deep_columns)(self.inputs)
		second_order_deep_columns = tf.keras.layers.Dense(64, activation=None)(second_order_deep_columns)
		second_order_deep_columns = tf.keras.layers.Reshape((-1, 64))(second_order_deep_columns)

		second_order_fm_feature = tf.keras.layers.Concatenate(axis=1)(
			second_order_cat_columns + [second_order_deep_columns])

		# build second _order_deep_feature
		deep_feature = tf.keras.layers.Flatten()(second_order_fm_feature)
		deep_feature = tf.keras.layers.Dense(32, activation='relu')(deep_feature)
		deep_feature = tf.keras.layers.Dense(16, activation='relu')(deep_feature)

		# build second order fm feature
		second_order_sum_feature = ReduceLayer(1)(second_order_fm_feature)
		second_order_sum_square_feature = tf.keras.layers.multiply([second_order_sum_feature, second_order_sum_feature])
		second_order_square_feature = tf.keras.layers.multiply([second_order_fm_feature, second_order_fm_feature])
		second_order_square_sum_feature = ReduceLayer(1)(second_order_square_feature)
		## second_order_fm_feature
		second_order_fm_feature = tf.keras.layers.subtract(
			[second_order_sum_square_feature, second_order_square_sum_feature])

		concatenated_outputs = tf.keras.layers.Concatenate(axis=1)(
			[first_order_feature, second_order_fm_feature, deep_feature])
		output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concatenated_outputs)

		model = tf.keras.Model(self.inputs, output_layer)
		return model


if __name__ == "__main__":
	# 构造模型
	# define input for keras model
	inputs = {
		'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
		'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
		'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
		'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
		'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
		'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
		'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

		'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
		'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
		'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),

		'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
		'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
		'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
		'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
		'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
		'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
		'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
		'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
	}
	# genre features vocabulary
	genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
	               'Sci-Fi', 'Drama', 'Thriller',
	               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

	movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
	user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
	user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1",
	                                                                           vocabulary_list=genre_vocab)
	item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1",
	                                                                           vocabulary_list=genre_vocab)

	movie_ind_col = tf.feature_column.indicator_column(movie_col)  # movid id indicator columns
	user_ind_col = tf.feature_column.indicator_column(user_col)  # user id indicator columns
	user_genre_ind_col = tf.feature_column.indicator_column(user_genre_col)
	item_genre_ind_col = tf.feature_column.indicator_column(item_genre_col)

	# cat_columns
	cat_columns = [movie_ind_col, user_ind_col, user_genre_ind_col, item_genre_ind_col]

	# deep_columns
	deep_columns = [tf.feature_column.numeric_column('releaseYear'),
	                tf.feature_column.numeric_column('movieRatingCount'),
	                tf.feature_column.numeric_column('movieAvgRating'),
	                tf.feature_column.numeric_column('movieRatingStddev'),
	                tf.feature_column.numeric_column('userRatingCount'),
	                tf.feature_column.numeric_column('userAvgRating'),
	                tf.feature_column.numeric_column('userRatingStddev')]

	movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
	user_emb_col = tf.feature_column.embedding_column(user_col, 10)
	user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, 10)
	item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, 10)

	embedding_column = [tf.keras.layers.DenseFeatures([item_genre_emb_col])(inputs),
	                    tf.keras.layers.DenseFeatures([movie_emb_col])(inputs),
	                    tf.keras.layers.DenseFeatures([user_genre_emb_col])(inputs),
	                    tf.keras.layers.DenseFeatures([user_emb_col])(inputs)
	                    ]

	model = DeepFM(inputs, cat_columns, deep_columns, embedding_column)


	# load sample as tf dataset
	def get_dataset(file_path):
		dataset = tf.data.experimental.make_csv_dataset(
			file_path,
			batch_size=12,
			label_name='label',
			na_value="0",
			num_epochs=1,
			ignore_errors=True)
		return dataset


	# 准备数据集
	# Training samples path, change to your local path
	training_samples_file_path = "../data/trainingSamples.csv"
	# Test samples path, change to your local path
	test_samples_file_path = "../data/testSamples.csv"
	# split as test dataset and training dataset
	train_dataset = get_dataset(training_samples_file_path)
	test_dataset = get_dataset(test_samples_file_path)

	model.compile_model(0.002)
	model.train(train_dataset, 2, val_data=test_dataset, model_file_path="../ckpt/deepFM_test")
