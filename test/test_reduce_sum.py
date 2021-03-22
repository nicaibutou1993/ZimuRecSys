import numpy as np

import tensorflow as tf
a = tf.constant([0,1,2,1,3,4,5,2,3,4,5,0],shape=[2,1,6])
b =tf.constant([1,3,2,3,1,2],shape=[6,1])

tensordot = tf.tensordot(a, b, axes=(2,0))


_xl = tf.matmul(a, tensordot, transpose_a=True)

_x2 = tf.add(_xl, b)

print("a的shape:",a.shape)
print("b的shape:",b.shape)
print("a的值:",tensordot)
print("a的值:",_xl)
print("a的值:",_x2)
#
# print(sort)
# print(v.sort(key = a) )

