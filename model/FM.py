import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model.BasicModel import BasicModel
K = tf.keras.backend


# 自定义FM的二阶交叉层
class FMLayer(tf.keras.layers.Layer):
	def __init__(self, input_dim, output_dim=4, **kwargs):
		self.input_dim = input_dim
		self.output_dim = output_dim
		super(FMLayer, self).__init__(**kwargs)

	# 初始化训练权重
	def build(self, input_shape):
		self.kernel = self.add_weight(name='kernel',
		                              shape=(self.input_dim, self.output_dim),
		                              initializer='glorot_uniform',
		                              trainable=True)
		super(FMLayer, self).build(input_shape)

	# 自定义FM的二阶交叉项的计算公式
	def call(self, x):
		a = K.pow(K.dot(x, self.kernel), 2)
		b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
		return K.sum(a - b, 1, keepdims=True) * 0.5

	# 输出的尺寸大小
	def compute_output_shape(self, input_shape):
		return input_shape[0], self.output_dim


class FM(BasicModel):
	def __init__(self, inputs, feature_columns, latent_feature_dim):
		self.inputs = inputs
		self.feature_columns = feature_columns
		self.latent_feature_dim = latent_feature_dim
		self.model = self.build_model()

	# 实现FM算法
	def build_model(self):
		inputs_Dense = tf.keras.layers.DenseFeatures(self.feature_columns)(self.inputs)
		# 线性回归
		liner = tf.keras.layers.Dense(units=1,
		                              bias_regularizer=tf.keras.regularizers.l2(0.01),
		                              kernel_regularizer=tf.keras.regularizers.l1(0.02),
		                              )(inputs_Dense)
		# FM的二阶交叉项
		cross = FMLayer(inputs_Dense.shape[1], output_dim=self.latent_feature_dim)(inputs_Dense)
		# 获得FM模型（线性回归 + FM的二阶交叉项）
		add = tf.keras.layers.Add()([liner, cross])
		predictions = tf.keras.layers.Activation('sigmoid')(add)
		model = tf.keras.Model(self.inputs, predictions)
		return model