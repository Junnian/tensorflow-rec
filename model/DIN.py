# -*- conding:utf-8 -*-
import tensorflow as tf
from model.BasicModel import BasicModel

class ActivateUnit(tf.keras.layers.Layer):
	def __init__(self, hidden_dim, embedding_size, **kwargs):
		super(ActivateUnit, self).__init__()
		self.hidden_dim = hidden_dim
		self.embedding_size = embedding_size

	def build(self, input_shape):
		pass

	def call(self, input, **kwargs):
		activation_unit = tf.keras.layers.Dense(32)(input)
		activation_unit = tf.keras.layers.PReLU()(activation_unit)
		activation_unit = tf.keras.layers.Dense(1, activation='sigmoid')(activation_unit)
		activation_unit = tf.keras.layers.Flatten()(activation_unit)
		activation_unit = tf.keras.layers.RepeatVector(self.embedding_size )(activation_unit)
		activation_unit = tf.keras.layers.Permute((2, 1))(activation_unit)
		return activation_unit

class DIN(BasicModel):
	def __init__(self, inputs, user_profile_columns, context_features_columns, candidate_columns, behaviors_columns,
	             item_nums, item_embedding_dim):
		self.inputs = inputs
		self.user_profile_columns = user_profile_columns
		self.context_features_columns = context_features_columns
		self.candidate_columns = candidate_columns
		self.behaviors_columns = behaviors_columns
		self.embedding_size = item_embedding_dim

		self.item_emb_layer = tf.keras.layers.Embedding(item_nums, item_embedding_dim, mask_zero=True)
		self.activation_unit_layer = ActivateUnit(32, item_embedding_dim)
		self.model = self.build_model()

	def build_model(self):
		# 用户特征编码
		user_profile_layer = tf.keras.layers.DenseFeatures(self.user_profile_columns)(self.inputs)

		# 上下文特征编码
		context_features_layer = tf.keras.layers.DenseFeatures(self.context_features_columns)(self.inputs)

		# 候选物料编码
		candidate_emb_layer = tf.keras.layers.DenseFeatures(self.candidate_columns)(self.inputs)
		candidate_emb_layer = self.item_emb_layer(candidate_emb_layer)
		candidate_emb_layer = tf.squeeze(candidate_emb_layer, axis=1)

		# DIN层
		repeated_candidate_emb_layer = tf.keras.layers.RepeatVector(len(self.behaviors_columns))(candidate_emb_layer)

		user_behaviors_emb_layer = tf.keras.layers.DenseFeatures(self.behaviors_columns)(self.inputs)
		user_behaviors_emb_layer = self.item_emb_layer(user_behaviors_emb_layer)
		activation_sub_layer = tf.keras.layers.Subtract()([user_behaviors_emb_layer,
		                                                   repeated_candidate_emb_layer])  # element-wise sub
		activation_product_layer = tf.keras.layers.Multiply()([user_behaviors_emb_layer,
		                                                       repeated_candidate_emb_layer])  # element-wise product
		## DIN层输入
		activation_all = tf.keras.layers.concatenate([activation_sub_layer, user_behaviors_emb_layer,
		                                              repeated_candidate_emb_layer, activation_product_layer], axis=-1)

		activation_unit = tf.keras.layers.Dense(32)(activation_all)
		activation_unit = tf.keras.layers.PReLU()(activation_unit)
		activation_unit = tf.keras.layers.Dense(1, activation='sigmoid')(activation_unit)
		activation_unit = tf.keras.layers.Flatten()(activation_unit)
		activation_unit = tf.keras.layers.RepeatVector(self.embedding_size)(activation_unit)
		activation_weight = tf.keras.layers.Permute((2, 1))(activation_unit)
		
		# activation_weight = self.activation_unit_layer(activation_all)

		activation_output = tf.keras.layers.Multiply()([user_behaviors_emb_layer, activation_weight])
		user_behaviors_pooled_layers = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(activation_output)

		# 输出层
		concat_layer = tf.keras.layers.concatenate([user_profile_layer, user_behaviors_pooled_layers,
	                                                  candidate_emb_layer, context_features_layer])
		output_layer = tf.keras.layers.Dense(128)(concat_layer)
		output_layer = tf.keras.layers.PReLU()(output_layer)
		output_layer = tf.keras.layers.Dense(64)(output_layer)
		output_layer = tf.keras.layers.PReLU()(output_layer)
		output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)

		model = tf.keras.Model(self.inputs, output_layer)
		return model

if __name__ == "__main__":
	EMBEDDING_SIZE = 10
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
		'userRatedMovie2': tf.keras.layers.Input(name='userRatedMovie2', shape=(), dtype='int32'),
		'userRatedMovie3': tf.keras.layers.Input(name='userRatedMovie3', shape=(), dtype='int32'),
		'userRatedMovie4': tf.keras.layers.Input(name='userRatedMovie4', shape=(), dtype='int32'),
		'userRatedMovie5': tf.keras.layers.Input(name='userRatedMovie5', shape=(), dtype='int32'),

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

	# user profile columns
	user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
	user_emb_col = tf.feature_column.embedding_column(user_col, EMBEDDING_SIZE)

	# user genre embedding feature
	user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1",
	                                                                           vocabulary_list=genre_vocab)
	user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, EMBEDDING_SIZE)
	user_profile_columns = [
	    user_emb_col,
	    user_genre_emb_col,
	    tf.feature_column.numeric_column('userRatingCount'),
	    tf.feature_column.numeric_column('userAvgRating'),
	    tf.feature_column.numeric_column('userRatingStddev'),
	]

	# context_features
	# item genre embedding feature
	item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1",
	                                                                           vocabulary_list=genre_vocab)
	item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, EMBEDDING_SIZE)
	context_features_columns = [
		item_genre_emb_col,
		tf.feature_column.numeric_column('releaseYear'),
		tf.feature_column.numeric_column('movieRatingCount'),
		tf.feature_column.numeric_column('movieAvgRating'),
		tf.feature_column.numeric_column('movieRatingStddev'),
	]

	#
	behaviors_columns = [
		tf.feature_column.numeric_column(key='userRatedMovie1', default_value=0),
		tf.feature_column.numeric_column(key='userRatedMovie2', default_value=0),
		tf.feature_column.numeric_column(key='userRatedMovie3', default_value=0),
		tf.feature_column.numeric_column(key='userRatedMovie4', default_value=0),
		tf.feature_column.numeric_column(key='userRatedMovie5', default_value=0),
	]

	candidate_columns = [ tf.feature_column.numeric_column(key='movieId', default_value=0),   ]

	model = DIN(inputs, user_profile_columns, context_features_columns, candidate_columns, behaviors_columns,
	             item_nums = 10001, item_embedding_dim=EMBEDDING_SIZE)


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
	model.train(train_dataset, 2, val_data=test_dataset, model_file_path="../ckpt/DUN_test")







