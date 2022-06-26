import tensorflow as tf
from model.BasicModel import BasicModel
K = tf.keras.backend

class EmbeddingMLP(BasicModel):
	def __init__(self, inputs, feature_columns, hidden_dim):
		self.inputs = inputs
		self.feature_columns = feature_columns
		self.hidden_dim = hidden_dim
		self.model = self.build_model()

	def build_model(self):
		inputs_Dense = tf.keras.layers.DenseFeatures(self.feature_columns)(self.inputs)
		# 线性回归
		hidden_layer = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs_Dense)
		hidden_layer =tf.keras.layers.Dense(self.hidden_dim, activation='relu')(hidden_layer)
		predictions =tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer)
		model = tf.keras.Model(self.inputs, predictions)
		return model
