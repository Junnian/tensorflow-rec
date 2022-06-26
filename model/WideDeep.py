import tensorflow as tf
from model.BasicModel import BasicModel


class WideDeep(BasicModel):
	def __init__(self, inputs, deep_feature_columns, crossed_feature_columns, hidden_dim):
		self.inputs = inputs
		self.deep_feature_columns = deep_feature_columns
		self.crossed_feature_columns = crossed_feature_columns
		self.hidden_dim = hidden_dim
		self.model = self.build_model()

	def build_model(self):
		deep = tf.keras.layers.DenseFeatures(self.deep_feature_columns)(self.inputs)
		deep = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(deep)
		deep = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(deep)
		# wide part for cross feature
		wide = tf.keras.layers.DenseFeatures(self.crossed_feature_columns)(self.inputs)
		both = tf.keras.layers.concatenate([deep, wide])
		output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(both)
		model = tf.keras.Model(self.inputs, output_layer)
		return model
