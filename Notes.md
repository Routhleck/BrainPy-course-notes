[TOC]

# 神经计算建模简介

## 计算神经科学的背景与使命

计算神经科学是**脑科学**对**类脑智能**的**桥梁**

### 两大目标

- 用计算建模的方法来阐明大脑功能的计算原理
- 发展类脑智能的模型和算法

### Prehistory

- 1907 LIF model 
  神经计算的本质
- 1950s HH model 
  电位定量化模型 最fundamental的
- 1960s Roll's cable equation 
  描述信号在轴突和树突怎么传递
- 1970s Amari, Wilson, Cowan et al.
  现今建模的基础
- 1982 Hopfield model(Amari-Hopfield model)
  引入物理学技术，吸引子模型
- 1988 Sejnowski et al. "Computational Neuroscience"(science)
  提出计算神经科学概念

**现在的计算神经科学对应于物理学的第谷-伽利略时代，对大脑工作原理还缺乏清晰的理论**

### Three levels of Brain Science

![image-20230823105226568](Notes.assets/image-20230823105226568.png)

- 大脑做什么
  Computational theory 
  -> Psychology & Cognitive Science
  -> Human-like Cognitive function
- 大脑怎么做
  Representation & Algorithm 
  -> Computational Neuroscience
  -> Brain-inspired model & algorithm
- 大脑怎么实现
  Implementation
  -> Neuroscience
  -> Neuromorphic computing

### Mission of Computational Neuroscience

> What I can not build a computational model, I do not understand

## 神经计算建模的目标与挑战

### Limitation of Deep Learning

- 不擅长对抗样本
- 对图像的理解有限

![image-20230823105836259](Notes.assets/image-20230823105836259.png)

### Brain is for Processing Dynamical Information

**We never "see" a static image**

![image-20230823105918336](Notes.assets/image-20230823105918336.png)

### The missing link

a computational model of higher cognitive functior

![image-20230823110617639](Notes.assets/image-20230823110617639.png)

现在只是做的**局部**的网络，没有一个成功的模型，能**从神经元出发构建网络，到系统层面上**

**原因**: 因为神经科学底层数据的缺失，可以考虑数据驱动、大数据的方式来加快发展

## 神经计算建模的工具

> 工欲行其事，必先利其器
> We need "PyTorch/TensorFlow" in Computational Neuroscience!

### Challenges in neural modelling

有不同的尺度

- Mutiple-scale
- Large-scale
- Multiple purposes

![image-20230823111212460](Notes.assets/image-20230823111212460.png)

> The modeling targets and methods are extremely complex, and we need a general framework.

### Limitations of Existing Brain Simulators

现今的框架不能满足以上

![image-20230823111509523](Notes.assets/image-20230823111509523.png)

### What are needed for a brain simulator

1. Efficiency
   High-speed simulation on parallel computing devices, etc.
2. Integration
   Integrated modeling of simulation, training, and analysis
3. Flexibility
   New models at all scales can be accommodated
4. Extensibility
   Extensible to new modeling methods(machine learning)

需要新的范式

### Our solution: BrainPy

4 levels

![image-20230823111903456](Notes.assets/image-20230823111903456.png)

## 神经计算建模举例

### Image understanding: an ill-posed problem

Image Understanding = image segmentation + image object recognition

> Chicken vs. Egg dilemma
>
> - Without segmentation, how to recognize
> - Without recognition, how to segment

**The solution of brain:** Analysis-by-synthesis 猜测与验证方法

### Reverse Hierarchy Theory

人的感知是整体到局部

### Two pathways for visual information processing

![image-20230823114517888](Notes.assets/image-20230823114517888.png)

### Key Computational Issues for Global-to-local Neural Information Processing

- What are global and local features
- How to rapidly extract global features
- How to generate global hypotheses
- How to implement from global to local processing
- The interplay between global and local features
- Others

#### How to extract global features

**Global first = Topology first**(大范围首先，陈霖)
视觉系统更敏感于拓扑性质的差异

> DNNs has difficulty to recognize topology

**A retina-SC network for topology detection**

视网膜到上丘的检测，Gap junction coupling ...

### A Model for Motion Pattern Recognition

Reservoir Module
Decision-making Module

### How to generate "global" hypotheses in the representation space

Attractor neural network

![image-20230823115853980](Notes.assets/image-20230823115853980.png)

Levy Flight in Animal Behaviors

![image-20230823120000911](Notes.assets/image-20230823120000911.png)

### How to process information from global to local

Push-pull Feedback

A hierarchical Hopfield Model

### Interplay between global and local features

A two-pathway model for object recognition

![image-20230823120750349](Notes.assets/image-20230823120750349.png)

Modeling visual masking 可以用two-pathway很好解释

# Programming basics

## Python Basics

### Values

- Boolean
- String
- Integer
- Float
- ...

### Keywords

Not allowed to use keywords, they define structure and rules of a language.

```python
help("keywords")
```

### Operators

数据之间的操作

#### For Integers and Floats

```python
a=5
b=3
# addition +
print("a+b=",atb)
# subtraction -
print("a-b=",a-b)
# multiplication *
print("axb="a*b)
# division /
print("a/b=",a/b)
# power **
print("a**b=",a**b)
```

#### Booleans

```python
#Boolean experssions
# equals: ==
print("5==5",5==5) 
# do not equal: !=
print("5!-5",5!=5)
# greater than: >
print("5>5",5>5)
# greater than or equal: >=
print("5>=5”5>=5)
```

```python
# logica operators
print("True and False:", True and False)
print("True or False:", True or False)
print("not False:", not False)
```

### Modules

Not all functionality available comes automatically when starting python.

```python
import match
import numpy as np
print(math.pi)
print(np.pi)

from numpy import pi
print(pi)

from numpy import *
print(pi)
```

### Control statements

#### If

```python
a = 5
# In Python, blocks of code are defined using indentation.
if a == 5:
	print("ok")
```

> ok

#### For

```python
# range(5) means a list with integers, 0, 1, 2, 3, 4
for i in range(5):
    print(i)
```

> 0
> 1
> 2
> 3
> 4

#### While

```python
i = 1
while i <= 100:
    print(i**3)
    i += i**3 # a += b is short for a = a+b
```

> 1
> 8
> 1000

### Functions

- Functions are used to abstract components of a program.
- Much like a mathematical function, they take some input and then find the result. start a function definition with a keyword def
- Then comes the function name, with arguments in braces, and then a colon.

```python
def func(args1, args2):
    pass
```

### Data types

#### List

- Group variables together
- Specific order
- Access item with brankets: [ ]
- List can be sliced
- List can be multiplied
- List can be added
- Lists are mutable
- Copying a list

```python
myList = [0, 1, 2, 0,"name"]
print("myList[0]:", myList[0])
print("myList[1]:", myList[1])
print("myList[3]:", myList[3])
print("myList[-1]:", myList[-1])
print("myList[-2]:", myList[-2])
```

> myList[0]: 0
> myList[1]: 1
> myList[3]: name
> myList[-1]: name
> myList[-2]: 2.0

```python
myList = [0, 1.0, "hello"]
print("myList[0:2]:", mylist[0:2])
print("myList*2:", myList*2)
myList2 = [2,"yes"]
print("myList+myList2:", myList+myList2)
```

> myList[0:2]: [0，1.0]
> myList*2: [0，1.0， hello'，0，1.0， hello']
> myList+myList2: [0，1.0，'hello'，2，yes']

#### tuple

Tuples are immutable.

#### dictionary

A dictionary is a collection of key-value pairs

```python
d = {}
d[1] = 2
d["a"] = 3

print("d: ", d)

c = {1:2, "a":3}
print("c: ", c)
print("c[1]: ", c[1])
```

> d: {1: 2, 'a': 3}
> c: {1: 2, 'a': 3}
> c[1]: 2

### Class

In Python, everything is an object. Classes are objects, instances of
classes are objects, modules are objects, and functions are objects.

 	1.  a **type**
 	2.  an internal **data representation** (primitive or composite)
 	3.  a set of procedures for **interaction** with the object

**a simple example**

```python
# define class

class Linear():
    pass

# instantiate object
layer1 = Linear()

print(layer1)
```

> `<__main__.Linear object at 0x7f88ad6c61d0>`

#### Initializing an object

```python
# define class
class Linear():
    # It refers to the object (instance) itself
    def __init__(self, n_input):
        self.n_input = n_input

layer1 = Linear(100)
layer2 = Linear(1000)

print("layer1 : ", layer1.n_input)
print("layer2 : ", layer2.n_input)
```

> layer1 : 100
> layer2 : 1000

#### Class has methods (similar to functions)

```python
# define class
class Linear():
    ### It refers to the the object (instance) itself
    def __init__(self, n_input, n_output):
	    self.n_input = n_input
        self.n_output = n_output
	def compute n params(self):
        num_params = self.n_input * self.n_output
        return num_params
layerl = Linear(10,100)
print(layerl.compute_n_params())
```

> 1000

## NumPy Basic

### Numpy Introduction

- Fundamental package for scientific computing with Python
- N-dimensional array object
- Linear algebra, frontier transform, random number capacities
- Building block for other packages (e.g. Scipy)

### Array

- Arrays are mutable
- Arrays attributes
- ...

```python
A = np.zeros((2, 2))
print(A)
```

> [[0. 0.]
> 	[0. 0.]]

```python
a.ndim		# 2 dimension
a.shape		# (2, 5) shape of array
a.size		# 10 $ of elements
a.T			# transpose
a.dtype		# data type
```

#### Array broadcasting

When operating on two arrays, numpy compares shapes. Two dimensions are compatible when
1. They are of equal size
2. One of them is 1

![image-20230823143622229](Notes.assets/image-20230823143622229.png)

### Vector operations

- Inner product
- Outer product
- Dot product (matrix multiplication)

```python
u = [1, 2, 3]
v = [1, 1, 1]

np.inner(u, v)
np.outer(u, v)
np.dot(u, v)
```

> 6
> array([[1, 1, 1],
> 			[2, 2, 2],
> 			[3, 3, 3]])
> 6

### Matrix operations

- `np.ones`
- `.T`
- `np.dot`
- `np.eye`
- `np.trace`
- `np.row_stack`
- `np.column_stack`

### Operations along axes

```python
a = np.ones((2, 3))
print(a)

a.sum()

a.sum(axis=0)

a.cumsum()

a.cumsum(axis=0)
```

### Slicing arrays

```python
a = np.random.random((2, 3))
print(a)

a[0,:] 	# first row, all columns
a[0:2] 	# first and second rows, al columns
a[:,1:3]# all rows, second and third columns
```

### Reshape

```python
a = np.ones((10,1))
a.reshape(2,5)
```

### Linear algebra

```python
qr				# Computes the QR decomposition
cholesky		# Computes the Cholesky decomposition
inv(A)			# Inverse
solve(A,b)		# Solves Ax = b for A full rank
lstsq(A,b)		# Solves arg minx //Ax - b//2
eig(A)			# Eigenvalue decomposition
eigvals(A)		# Computes eigenvalues
svd(A，full)		# Sinqular value decomposition
pinv(A)			# Computes pseudo-inverse of A
```

### Fourier transform

```python
import numpy.fft
fft		# 1-dimensional DFT
fft2	# 2-dimensional DFT
fftn	# N-dimensional DFT
ifft	# 1-dimensional inverse DFT (etc.)
rfft	# Real DFT (1-dim)
```

### Random sampling

```python
import numpy.random
rand(d0, d1, ..., dn)		# Random values in a given shape
randn(d0, d1, ..., dn)		# Random standard normal
randint(lo, hi, size)		# Random integers [lo hi)
choice(a, size, repl, p)	# Sample from a
shuffle(a)					# Permutation (in-place)
permutation(a)				# Permutation (new array)
```

### Distributions in random

```python
import numpy.random
beta
binomial
chisquare
exponential
dirichlet
gamma
laplace
lognormal
...
```

### Scipy

- `SciPy` is a library of algorithms and mathematical tools built to work with `NumPy ` arrays.
- `scipy.linalg linear algebra`
- `scipy.stats statistics`
- `scipy.optimize optimization`
- `scipy.sparse sparse matrices`
- `scipy.signal signal processing`
- etc.

## BrainPy introduction

### Modeling demands

- Large-scale
- Multi-scale
- Methods

### BrainPy Architecture

- Infrastructure
- Functions
- Just-in-time compilation
- Devices

![image-20230823145349681](Notes.assets/image-20230823145349681.png)

### Main features

#### Dense operators

- Compatible with `NumPy`, `TensorFlow`, `PyTorch` and other dense matrix operator syntax.
- Users do not need to learn and get started programming directly.

#### Dedicated operatorsq

- Applies brain dynamics sparse connectivity properties with event-driven computational features.
- Reduce the complexity of brain dynamics simulations by several orders of magnitude.

#### Numerical Integrators

- Ordinary differential equations: brainpy.odeint
- Stochastic differential equations: brainpy.sdeint
- Fractional differential equations: brainpy.fdeint
- Delayed differential equations

#### Modular and composable

从微观到宏观

**brainpy.DynamicalSystem**

![image-20230823151159786](Notes.assets/image-20230823151159786.png)

#### JIT of object-oriented

BrainPy provides object-oriented transformations:

- `brainpy.math.jit`
- `brainpy.math.grad`
- `brainpy.math.for_loop`
- `brainpy.math.ifelse`

## BrainPy Programming Basics

### Just-in-Time compilation

Just In Time Compilation (JIT, or Dynamic Translation), is compilation that is being done during the execution of a program.

JIT compilation attempts to use **the benefits of both**. While the interpreted program is being run, the JIT compiler determines the most frequently used code and compiles it to machine code.

The advantages of a JIT are due to the fact that since the compilation takes place in run time, a JIT compiler has access to dynamic runtime information enabling it to make better optimizations (such as inlining functions).

```python
def gelu(x):
    sqrt = bm.sqrt(2 / bm.pi)
    cdf = 0.5 * (1.0 + bm.tanh(sqrt * (x + 0.044715 * (x ** 3))))
    y = x *cdf
    return y

>>> gelu_jit = bm.jit(gelu) # 使用JIT
```

### Object-oriented JIT compilation

- The class object must be inherited from brainpy.BrainPyObject, the base class of BrainPy, whose methods will be automatically JIT compiled.
- All time-dependent variables must be defined as brainpy.math.Variable.

```python
class LogisticRegression(bp.BrainPyObject):
    def __init__(self, dimension):
        super(LogisticRegression, self).__init__()
        
        # parameters
        self.dimension = dimension
        
        # variables
        self.w = bm.Variable(2.0 * bm.ones(dimension) - 1.3)
        
	def __call__(self, X, Y):
        u = bm.dot(((1.0 / (1.0 + bm.exp(-Y * bm.dot(X, self.w))) - 1.0) * Y), X)
        self.w.value = self.w - u # in-place update
```

**ExampleL Run a neuron model**

```python
model = bp.neurons.HH(1000) #一共1000个神经元
runner = bp.DSRunner(target=model, inputs=('input', 10.)) # jit默认为True
runner(duration=1000, eval_time=True) #模拟 1000ms
```

禁用JIT来debug

### Data operations

#### Array

等价于`numpy`的`array`

#### BrainPy arrays & JAX arrays

```python
t1 = bm.arange(3)
print(t1)
print(t1.value)
```

> JaxArray([0, 1, 2], dtype=int32)
> DeviceArray([0, 1, 2], dtype=int32)

#### Variables

Arrays that are not marked as dynamic variables will be JIT-compiled as static arrays, and modifications to static arrays will not be valid in the JIT compilation environment.

```python
t = bm.arange(4)
v = bm.Variable(t)
print(v)

print(v.value)
```

> Variable([0, 1, 2, 3], dtype=int32)
> DeviceArray([0, 1, 2, 3], dtype=int32)

### Variables

**In-place updating** 就地更新

#### Indexing and slicing

- Indexing: `v[i] = a` or `v[(1, 3)] = c`
- Slicing: `v[i:j] = b`
- Slicing all values `v[:] = d`, `v[...] = e`

#### Augmented assignment

- add
- subtract
- divide
- multiply
- floor divide
- modulo
- power
- and
- or
- xor
- left shift
- right shift

#### Value assignment

```python
v.value = bm.arange(10)
check_no_change(v)
```

#### Update assignment

```python
v.update(bm.random.randint(0, 20, size=10))
```

### Control flows

#### If-else

`brainpy.math.where`

```python
a = 1.
bm.where(a < 0, 0., 1.)
```

> DeviceArray(1., dtype=float32, weak_type=True)

`brainpy.math.ifelse`

```python
def ifelse(condition, branches, operands):
	true_fun, false_fun = branches
    if condition:
        return true_fun(operands)
    else:
        return false_fun(operands)
```

#### For loop

```python
import brainpy.math
hist_of_out_vars = brainpy.math.for_loop(body_fun, operands)
```

#### While loop

```python
i = bm.Variable(bm.zeros(1))
counter = bm.Variable(bm.zeros(1))

def cond_f():
    return i[0] < 10

def body_f():
    i.value += 1
    counter.value += i

bm.while_loop(body_f, cond_f, operands=())

print(counter, i)
```