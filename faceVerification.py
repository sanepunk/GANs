# from tkinter.font import names
import torchvision
import tensorflow as tf
import jax.numpy as jnp
from flax import linen as nn
import jax
from typing import Union
import torch

class Liveliness(nn.Module):
	@nn.compact
	def __call__(self, inputs: Union[jnp.ndarray, tf.TensorArray, tf.Tensor, torch.Tensor, torch.FloatTensor], training: bool = True):
		x1_skip = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME', name='Conv_0(skipped_input_0)')(inputs)
		x1 = nn.relu(x1_skip)
		x1 = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME', name='Conv_1')(x1)
		x1 = nn.relu(x1)
		# print(x1_skip.shape, x1.shape)
		x_skip_conn_1 = nn.max_pool((nn.BatchNorm(use_running_average=not training,
												  name='Batch_Norm(Conv_1 + skipped_input_0)')(nn.gelu(x1 + x1_skip,
																									  approximate=True))), window_shape=(2, 2), strides=(2, 2))
		x2_skip = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='Conv_2(skipped_input_1)')(x_skip_conn_1)
		x2 = nn.relu(x2_skip)
		x2 = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='Conv_3')(x2)
		x2 = nn.relu(x2)
		x_skip_conn_2 = nn.max_pool(nn.BatchNorm(use_running_average=not training,
												 name='Batch_Norm(Conv_3 + skipped_input_1)')(nn.gelu(x2_skip + x2,
																									 approximate=True)), window_shape=(2, 2), strides=(2, 2))
		x3_skip = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', name='Conv_4(skipped_input_2)')(x_skip_conn_2)
		x3 = nn.relu(x3_skip)
		x3 = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', name='Conv_5')(x3)
		x3 = nn.relu(x3)
		x_skip_conn_3 = nn.max_pool(nn.BatchNorm(use_running_average=not training,
												 name='Batch_Norm(Conv_5 + skipped_input_2)')(nn.gelu(x3_skip + x3,
																									 approximate=True)), window_shape=(2, 2), strides=(2, 2))
		x4_skip = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='Conv_6(skipped_input_3)')(x_skip_conn_3)
		x4 = nn.relu(x4_skip)
		x4 = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='Conv_7')(x4)
		x4 = nn.relu(x4)
		x_skip_conn_4 = nn.max_pool(nn.BatchNorm(use_running_average=not training,
												 name='Batch_Norm(Conv_7 + skipped_input_3)')(nn.gelu(x4_skip + x4,
																									 approximate=True)), window_shape=(2, 2), strides=(2, 2))
		x5 = x_skip_conn_4.reshape((x_skip_conn_4.shape[0], -1))
		# print(x5.shape)
		x5 = nn.leaky_relu(nn.Dense(features=1638)(x5))
		x5 = nn.leaky_relu(nn.Dense(features=10, name='O/P Layer')(x5))
		return x5


model = Liveliness()
x = jax.random.normal(jax.random.PRNGKey(42), (5, 128, 128, 3))
params = model.init(jax.random.PRNGKey(0), jax.random.normal(jax.random.PRNGKey(10), (1, 128, 128, 3)) * 1e-4, training = False)['params']
# y, n = model.apply({'params': params}, x, mutable=['batch_stats'], training = True)
# print(model.tabulate(jax.random.PRNGKey(0), jax.random.normal(jax.random.PRNGKey(10), (1, 128, 128, 3)) * 1e-4, training = True))

similarity_search = torchvision.models.vgg16(pretrained=True)

i: int = 0
similarity_search = torch.nn.Sequential(*list(similarity_search.children())[:23])
for layer in similarity_search.children():
	# if i < 23:
	for param in layer.parameters():
		param.requires_grad = False
for layers in similarity_search.children():
	print(layers)
