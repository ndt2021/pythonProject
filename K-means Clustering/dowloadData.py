from __future__ import print_function
import numpy as np
from sklearn.datasets import fetch_openml

import os

# Lấy thư mục gốc của mô-đun
module_dir = os.path.dirname(__file__)

# Tạo đường dẫn tương đối tới thư mục 'data' trong thư mục gốc của mô-đun
data_dir = os.path.join(module_dir, 'data')

# Tải dữ liệu MNIST từ OpenML
mnist = fetch_openml('mnist_784', version=1, data_home=data_dir)

print("Shape of mnist data:", mnist.data.shape)

