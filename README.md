[![Python package](https://github.com/SermetPekin/micrograd/actions/workflows/python-package.yml/badge.svg?1)](https://github.com/SermetPekin/micrograd/actions/workflows/python-package.yml?1)
![](https://img.shields.io/badge/python-3.10+-blue.svg)


## Acknowledgment

This project is a fork of [micrograd](https://github.com/karpathy/micrograd) by [Andrej Karpathy](https://github.com/karpathy), which is licensed under the MIT License. 

The original micrograd library served as the foundation for this project, providing a simple and elegant implementation of an automatic differentiation engine. All credit for the original implementation and its inspiration goes to Andrej Karpathy.

This fork includes additional features, improvements, and extensions to the original project to enhance its functionality and adapt it for new use cases. Special thanks to Andrej Karpathy for sharing his work with the community and making this project possible.



# micrograd
```bash 
git clone https://github.com/SermetPekin/micrograd.git
# usage with uv package (just like poetry / pipvenv but faster to create env and install dependencies)
pip install uv 
uv venv 
source .venv\scripts\activate
uv pip install . 
# run example 
python ./example.py
# for tests
pytest -v 

```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b ** 3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e ** 2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}')  # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}')  # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}')  # prints 645.5773, i.e. the numerical value of dg/db
```


