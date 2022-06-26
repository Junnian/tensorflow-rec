import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

K = tf.keras.backend

class EmbeddingMLP:
	def __init__(self, inputs, feature_columns, hidden_dim):
		self.inputs = inputs
		self.feature_columns = feature_columns
		self.hidden_dim = hidden_dim
		self.model = self.build_model()

	# 实现FM算法
	def build_model(self):
		inputs_Dense = tf.keras.layers.DenseFeatures(self.feature_columns)(self.inputs)
		# 线性回归
		hidden_layer = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs_Dense)
		hidden_layer =tf.keras.layers.Dense(self.hidden_dim, activation='relu')(hidden_layer)
		predictions =tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer)
		model = tf.keras.Model(self.inputs, predictions)
		return model

	def compile_model(self, lr):
		self.model.compile(optimizer=tf.keras.optimizers.Adam(lr),
		                   loss=tf.keras.losses.binary_crossentropy,
		                   metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

	def train(self, data, epochs, val_data=None, patience=0, model_file_path=None):

		callbacks = []
		if patience:
			earlystop_callback = EarlyStopping(monitor="val_auc", patience=patience, verbose=0, mode='max')
			callbacks.append(earlystop_callback)

		if model_file_path:
			model_save_call_back = ModelCheckpoint(
				model_file_path, monitor='val_auc', save_best_only=True,
				save_weights_only=False, mode='max', save_freq="epoch", verbose=1)
			callbacks.append(model_save_call_back)

		self.model.fit(data,
		               epochs=epochs,
		               validation_data=val_data,
		               callbacks=callbacks)

	def predict(self, data):
		return self.model.predict(data)

	def save(self, save_path):
		self.model.save(save_path)

	def load(self, model_path):
		return tf.keras.models.load_model(model_path)
