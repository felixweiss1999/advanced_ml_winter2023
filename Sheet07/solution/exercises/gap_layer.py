import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D

# initialize some test data
A = np.random.randint(0, 10, (3, 4, 5, 6)).astype(float)

# initialize global average pooling layer
gap_layer = GlobalAveragePooling2D()

# TensorFlow's output
M_tf = gap_layer(A).numpy()

# NumPy's computation
M_np = A.mean(axis=(1,2))

print('#######################################')
print('# Comparison of TensorFlow and NumPy: #')
print('#######################################')
print('tf.keras.layers.GlobalAveragePooling2D:')
print(M_tf)
print('Average over height and width in NumPy:')
print(M_np)
