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

# Single Neuron Modeling: Conductance-Based Models

## Neuronal structure,ing potential, and equivalent circuits

### Neuronal structure

- Cell body/soma
- Axon
- Dendrites
- Synapses

![image-20230824100656397](Notes.assets/image-20230824100656397.png)

### Resting potential

Transport proteins for ions in neuron cell membranes:

- Ion channels: Na + channels, K + channels, … (gated/non-gated)
- Ion pumps: the Na + -K + pump

![image-20230824100812962](Notes.assets/image-20230824100812962.png)

离子浓度在胞内外的差异产生的电势差

- Ion concentration difference → chemical gradient → electrical gradient

- Nernst Equation:
  $$
  E=\dfrac{RT}{zF}\ln\dfrac{[\mathrm{ion}]_{\mathrm{out}}}{[\mathrm{ion}]_{\mathrm{in}}}
  $$
  
- Goldman-Hodgkin-Katz (GHK) Equation:
  $$
  V_m=\frac{RT}{F}\ln\left(\frac{P_{\mathrm{Na}}[\mathrm{Na}^+]_{\mathrm{out}}+P_{\mathrm{K}}[\mathrm{K}^+]_{\mathrm{out}}+P_{\mathrm{Cl}}[\mathrm{Cl}^-]_{\mathrm{in}}}{P_{\mathrm{Na}}[\mathrm{Na}^+]_{\mathrm{in}}+P_{\mathrm{K}}[\mathrm{K}^+]_{\mathrm{in}}+P_{\mathrm{Cl}}[\mathrm{Cl}^-]_{\mathrm{out}}}\right)
  $$

### Equivalent circuits

Components of an equivalent circuit:

- Battery
- Capacitor
- Resistor

![image-20230824101350048](Notes.assets/image-20230824101350048.png)

Considering the potassium channel **ONLY**:
$$
\begin{gathered}
0=I_{\mathrm{cap}}+I_{K}=c_{\mathrm{M}}{\frac{\mathrm{d}V_{\mathrm{M}}}{\mathrm{d}t}}+{\frac{V_{\mathrm{M}}-E_{\mathrm{K}}}{R_{\mathrm{K}}}}, \\
c_{\mathrm{M}}{\frac{\mathrm{d}V_{\mathrm{M}}}{\mathrm{d}t}}=-{\frac{V_{\mathrm{M}}-E_{\mathrm{K}}}{R_{\mathrm{K}}}}=-g_{\mathrm{K}}(V_{\mathrm{M}}-E_{\mathrm{K}}). 
\end{gathered}
$$
![image-20230824101433908](Notes.assets/image-20230824101433908.png)

**Considering the Na + , K + , and Cl - channels and the external current I(t):**

![image-20230824101648270](Notes.assets/image-20230824101648270.png)
$$
\begin{aligned}
\frac{I(t)}{A}& =c_{\mathrm{M}}{\frac{\mathrm{d}V_{\mathrm{M}}}{\mathrm{d}t}}+i_{\mathrm{ion}}  \\
\Rightarrow{{c_{\mathrm{M}}\frac{\mathrm{d}V_{\mathrm{M}}}{\mathrm{d}t}}}& =-g_{\mathrm{Cl}}(V_{\mathrm{M}}-E_{\mathrm{Cl}})-g_{\mathrm{K}}(V_{\mathrm{M}}-E_{\mathrm{K}})-g_{\mathrm{Na}}(V_{\mathrm{M}}-E_{\mathrm{Na}})+\frac{I(t)}{A} 
\end{aligned}
$$
Steady-state membrane potential given a constant current input I:
$$
\begin{array}{rcl}\Rightarrow&c_{M}\frac{\mathrm{d}V_{M}}{\mathrm{d}t}=-(g_{C1}+g_{K}+g_{Na})V_{M}+g_{C1}E_{C1}+g_{K}E_{K}+g_{Na}E_{Na}+\frac{I(t)}{A}\\\\V_{sS}=\frac{g_{CM}E_{C1}+g_{K}E_{K}+g_{Na}E_{Na}+I/A}{g_{C1}+g_{K}+g_{Na}}&
\xrightarrow{I=0}
&V_{sN,I=0}=E_{R}=\frac{g_{CC}E_{C1}+g_{K}E_{K}+g_{Na}E_{Na}}{g_{C1}+g_{K}+g_{Na}}\end{array}
$$

## Cable Theory & passive conduction

![image-20230824102017607](Notes.assets/image-20230824102017607.png)

Considering the axon as a long cylindrical cable:
$$
I_{\mathrm{cross}}(x,t)={I_{\mathrm{cross}}(x+\Delta x,t)}+I_{\mathrm{ion}}(x,t)+I_{\mathrm{cap}}(x,t)
$$

$$
V(x+\Delta x,t)-V(x,t)=-I_{\mathrm{cross}}(x,t)R_{\mathrm{L}}=-I_{\mathrm{cross}}(x,t)\frac{\Delta x}{\pi a^{2}}\rho_{\mathrm{L}} \\
{I_{\mathrm{cross}}(x,t)} =-\frac{\pi a^{2}}{\rho_{\mathrm{L}}}\frac{\partial V(x,t)}{\partial x}  \\
{I_{\mathrm{ion}}} =(2\pi a\Delta x)i_{\mathrm{ion}}  \\
I_{\mathrm{cap}}(x,t) =(2\pi a\Delta x)c_{\mathrm{M}}\frac{\partial V(x,t)}{\partial t} 
$$

-> 
$$
(2\pi a\Delta x)c_{\mathrm{M}}\frac{\partial V(x,t)}{\partial t}+(2\pi a\Delta x)i_{\mathrm{ion}}=\frac{\pi a^{2}}{\rho_{\mathrm{L}}}\frac{\partial V(x+\Delta x,t)}{\partial x}-\frac{\pi a^{2}}{\rho_{\mathrm{L}}}\frac{\partial V(x,t)}{\partial x}
$$
**Cable Equation**
$$
c_\mathrm{M}\frac{\partial V(x,t)}{\partial t}=\frac{a}{2\rho_\mathrm{L}}\frac{\partial^2V(x,t)}{\partial x^2}-i_\mathrm{ion}
$$
电流在通过长直导体时会泄露电流，如何记录膜电位，可以使用此方程来描述

**Passive conduction:** ion currents are caused by leaky channels exclusively
$$
i_{\mathrm{ion}}=V(x,t)/r_{\mathrm{M}}
$$
->
$$
\begin{aligned}c_\mathrm{M}\frac{\partial V(x,t)}{\partial t}&=\frac{a}{2\rho_\mathrm{L}}\frac{\partial^2V(x,t)}{\partial x^2}-\frac{V(x,t)}{r_\mathrm{M}}\\\\\tau\frac{\partial V(x,t)}{\partial t}&=\lambda^2\frac{\partial^2V(x,t)}{\partial x^2}-V(x,t)\quad\lambda=\sqrt{0.5ar_\mathrm{M}/\rho_\mathrm{L}}\end{aligned}
$$
没有动作电位，单纯通过电缆传输

![image-20230824102932665](Notes.assets/image-20230824102932665.png)

If a constant external current is applied to 𝑥 = 0  the steady-state membrane potential $𝑉_{ss}(𝑥)$ is
$$
\lambda^2\frac{\mathrm{d}^2V_{\mathrm{ss}}(x)}{\mathrm{d}x^2}-V_{\mathrm{ss}}(x)=0\longrightarrow V_{\mathrm{ss}}(x)=\frac{\lambda\rho_{\mathrm{L}}}{\pi a^2}I_0e^{-x/\lambda}
$$
电信号无衰减传播: 动作电位

## Action potential & active transport

Steps of an action potential:

- Depolarization
- Repolarization
- Hyperpolarization
- Resting

Characteristics:

- All-or-none
- Fixed shape
- Active electrical property

![image-20230824103322522](Notes.assets/image-20230824103322522.png)

How to simulate an action potential?
$$
\begin{aligned}
\frac{I(t)}{A}& =c_{\mathrm{M}}{\frac{\mathrm{d}V_{\mathrm{M}}}{\mathrm{d}t}}+i_{\mathrm{ion}}  \\
\Rightarrow\quad c_{\mathrm{M}}\frac{\mathrm{d}V_{\mathrm{M}}}{\mathrm{d}t}& =-g_{\mathrm{Cl}}(V_{\mathrm{M}}-E_{\mathrm{Cl}})-g_{\mathrm{K}}(V_{\mathrm{M}}-E_{\mathrm{K}})-g_{\mathrm{Na}}(V_{\mathrm{M}}-E_{\mathrm{Na}})+\frac{I(t)}{A} 
\end{aligned}
$$
离子通道的开闭会随着电压而变化，电导也随着电压而变化

Mechanism: voltage-gated ion channels

**HH建模思路：通过电导**

### Nodes of Ranvier

Saltatory conduction with a much higher speed and less energy consumption

两个郎飞结之间会有离子通道，既有被动传导，也有主动的防止衰减

![image-20230824104220106](Notes.assets/image-20230824104220106.png)

## The Hodgkin-Huxley Model

### Modeling of each ion channel

Modeling of each ion channel:
$$
g_m=\bar{g}_mm^x
$$
Modeling of each ion gate:
$$
\mathcal{C}\underset{}{\operatorname*{\overset{\alpha(\mathrm{V})}{\underset{\beta(\mathrm{V})}{\operatorname*{\longrightarrow}}}}\mathcal{O}}

\\
\Rightarrow
\begin{aligned}
\frac{\mathrm{d}m}{\mathrm{d}t}& =\alpha(V)(1-m)-\beta(V)m  \\
&=\frac{m_{\infty}(V)-m}{\tau_{m}(V)}
\end{aligned}

\\
\\

\begin{aligned}m_\infty(V)&=\frac{\alpha(V)}{\alpha(V)+\beta(V)}.\\\tau_m(V)&=\frac{1}{\alpha(V)+\beta(V)}\end{aligned}
$$

$$
\text{If}\ V\text{ is constant:}m(t)=m_\infty(V)+(m_0-m_\infty(V))\mathrm{e}^{-t/\tau_m(V)}
$$

### Voltage clamp

$$
\begin{aligned}
\frac{I(t)}{A}& =c_{\mathrm{M}}{\frac{\mathrm{d}V_{\mathrm{M}}}{\mathrm{d}t}}+i_{\mathrm{ion}}  \\
\Rightarrow\quad c_{\mathrm{M}}\frac{\mathrm{d}V_{\mathrm{M}}}{\mathrm{d}t}& =-g_{\mathrm{Cl}}(V_{\mathrm{M}}-E_{\mathrm{Cl}})-g_{\mathrm{K}}(V_{\mathrm{M}}-E_{\mathrm{K}})-g_{\mathrm{Na}}(V_{\mathrm{M}}-E_{\mathrm{Na}})+\frac{I(t)}{A} 
\end{aligned}
$$

- The membrane potential is kept constant
- The current from capacitors is excluded
- Currents must come from leaky/voltage-gated ion channels

$$
\begin{aligned}I_{\mathrm{cap}}&=c\frac{dV}{dt}=0\\I_{\mathrm{fb}}&=\quad i_{\mathrm{ion}}=g_{\mathrm{Na}}(V-E_{\mathrm{Na}})+g_{\mathrm{K}}(V-E_{\mathrm{K}})+g_{\mathrm{L}}(V-E_{\mathrm{L}})\end{aligned}
$$

只测量一个离子通道就可以很容易得到电导

![image-20230824111620056](Notes.assets/image-20230824111620056.png)

### Leaky channel

Hyperpolarization → the sodium and potassium channels are closed
$$
I_{\mathrm{fb}}=g_{\mathrm{Na}}(V-E_{\mathrm{Na}})+g_{\mathrm{K}}(V-E_{\mathrm{K}})+g_{\mathrm{L}}(V-E_{\mathrm{L}})
$$

$$
\Rightarrow I_{\mathrm{fb}}=g_L(V-E_L)
$$

$$
g_\mathrm{L}=0.3\mathrm{mS/cm}^2,E_\mathrm{L}=-54.4\mathrm{mV}
$$

#### Potassium and sodium channels

Potassium channels: Use choline to eliminate the inward current of Na +
Na + current: $I_{fb} - I_{K}$

![image-20230824112328953](Notes.assets/image-20230824112328953.png)

![image-20230824112333144](Notes.assets/image-20230824112333144.png)

转化速率和电导率两个因素

Potassium channels

- Resting state (gate closed)
- Activated state (gate open)

→ Activation gate: $g_{\mathrm{K}}=\bar{g}_{K}n^{x}$

Sodium channels

- Resting state (gate closed)
- Activated state (gate open)
- Inactivated state (gate blocked)

→ Activation gate + inactivation gate: $g_{\mathrm{Na}}=\bar{g}_\text{Na}m^3h$

![image-20230824113116329](Notes.assets/image-20230824113116329.png)

The gates of sodium channels

Modeling of each ion gate:
$$
\begin{aligned}
&\text{gk}&& =\bar{g}_{K}n^{x}  \\
&\text{gNa}&& =\bar{g}_{\mathrm{Na}}m^{3}h  \\
&\frac{\mathrm{d}n}{\mathrm{d}t}&& =\alpha_{n}(V)(1-n)-\beta_{n}(V)n  \\
&\frac{\mathrm{d}m}{\mathrm{d}t}&& =\alpha_{m}(V)(1-m)-\beta_{m}(V)m  \\
&\frac{\mathrm{d}h}{\mathrm{d}t}&& =\alpha_{h}(V)(1-h)-\beta_{h}(V)h 
\end{aligned}
$$

$$
\begin{aligned}
\frac{\mathrm{d}m}{\mathrm{d}t}& =\alpha(V)(1-m)-\beta(V)m  \\
&=\frac{m_{\infty}(V)-m}{\tau_{m}(V)}
\end{aligned}
$$

$$
\begin{aligned}m_\infty(V)&=\frac{\alpha(V)}{\alpha(V)+\beta(V)}\\\tau_m(V)&=\frac{1}{\alpha(V)+\beta(V)}\end{aligned}.
$$

$$
m(t)=m_\infty(V)+(m_0-m_\infty(V))\mathrm{e}^{-t/\tau_m(V)}
$$

### The Hodgkin-Huxley(HH) Model

$$
c_\mathrm{M}\frac{\mathrm{d}V_\mathrm{M}}{\mathrm{d}t}=-g_\mathrm{Cl}(V_\mathrm{M}-E_\mathrm{Cl})-g_\mathrm{K}(V_\mathrm{M}-E_\mathrm{K})-g_\mathrm{Na}(V_\mathrm{M}-E_\mathrm{Na})+\frac{I(t)}{A}
$$

本质是4个微分方程联立在一起
$$
\left\{\begin{aligned}&c\frac{\mathrm{d}V}{\mathrm{d}t}=-\bar{g}_\text{Na}m^3h(V-E_\text{Na})-\bar{g}_\text{K}n^4(V-E_\text{K})-\bar{g}_\text{L}(V-E_\text{L})+I_\text{ext},\\&\frac{\mathrm{d}n}{\mathrm{d}t}=\phi\left[\alpha_n(V)(1-n)-\beta_n(V)n\right]\\&\frac{\mathrm{d}m}{\mathrm{d}t}=\phi\left[\alpha_m(V)(1-m)-\beta_m(V)m\right],\\&\frac{\mathrm{d}h}{\mathrm{d}t}=\phi\left[\alpha_h(V)(1-h)-\beta_h(V)h\right],\end{aligned}\right.
$$

$$
\begin{aligned}\alpha_n(V)&=\frac{0.01(V+55)}{1-\exp\left(-\frac{V+55}{10}\right)},\quad\beta_n(V)&=0.125\exp\left(-\frac{V+65}{80}\right),\\\alpha_h(V)&=0.07\exp\left(-\frac{V+65}{20}\right),\quad\beta_n(V)&=\frac{1}{\left(\exp\left(-\frac{V+55}{10}\right)+1\right)},\\\alpha_m(V)&=\frac{0.1(V+40)}{1-\exp\left(-(V+40)/10\right)},\quad\beta_m(V)&=4\exp\left(-(V+65)/18\right).\end{aligned}
$$

$$
\phi=Q_{10}^{(T-T_{\mathrm{base}})/10}
$$

每一步符合生物学

![image-20230824113714178](Notes.assets/image-20230824113714178.png)

#### How to fit each gating variable?

**Fitting n:** $g_{\mathbf{K}}=\bar{g}_{K}n^{x}\quad m(t)=m_{\infty}(V)+(m_{0}-\color{red}{\boxed{m_{\infty}(V)}})\mathrm{e}^{-t/\pi_{m}(V)}$

→ $g_\mathrm{K}(V,t)=\bar{g}_\mathrm{K}\left[n_\infty(V)-(n_\infty(V)-n_0(V))\mathrm{e}^{-\frac{t}{\tau_n(V)}}\right]^x$

by $g_{\mathrm{K}\infty}=\bar{g}_{\mathrm{K}}n_{\infty}^{x},g_{\mathrm{K}0}=\bar{g}_{\mathrm{K}}n_{0}^{x}$

→ $g_{\mathrm{K}}(V,t)=\left[g_{\mathrm{K}\infty}^{1/x}-(g_{\mathrm{K}\infty}^{1/x}-g_{\mathrm{K}0}^{1/x})\mathrm{e}^{-\frac{t}{\tau_{n}(V)}}\right]^{x}$



![image-20230824114623467](Notes.assets/image-20230824114623467.png)

# Hodgkin-Huxley brain dynamics programming

## Dynamics Programming Basics

### Integrators

微分器

![image-20230824140806650](Notes.assets/image-20230824140806650.png)

**example**

FitzHugh-Nagumo equation
$$
\begin{aligned}\tau\dot{w}&=v+a-bw,\\\dot{v}&=v-\frac{\nu^3}{3}-w+I_{\mathrm{ext}}.\end{aligned}
$$

```python
@bp.odeint(method='Euler', dt=0.01)
def integral(V, w, t, Iext, a, b, tau):
    dw = (V + a - b * w) / tau
    dV = V - V * V * V / 3 - w + Iext
    return dV, dw
```

**JointEq**

In a dynamical system, there may be multiple variables that change dynamically over time. Sometimes these variables are interrelated, and updating one variable requires other variables as inputs. For better integration accuracy, we recommend that you use `brainpy.JointEq` to jointly solve interrelated differential equations.

```python
a, b = 0.02, 0.20
dV = lambda V, t, w, Iext: 0.04 * V * V + 5 * V + 140 - w + Iext	# 第一个方程
dw = lambda w, t, V: a * (b * V - w)								# 第二个方程
joint_eq = bp.JointEq(dV, dw)										# 联合微分方程
integral2 = bp.odeint(joint_eq, method='rk2')						# 定义该联合微分方程的数值积分方法
```

```python
# 声明积分运行器
runner = bp.integrators.IntegratorRunner(
	integral,
    monitors=['V']
    inits=dict(V=0., w=0.)
    args=dict(a=a, b=b, tau=tau, Iext=Iext),
    dt=0.01
)

# 使用积分运行器来进行模拟100ms，结合步长dt=0.01
runner.run(100.)

plt.plot(runner.mon.ts, runner.mon.V)
plt.show()
```

![image-20230824142019832](Notes.assets/image-20230824142019832.png)

### `DynamicalSystem`

BrainPy provides a generic `SynamicalSystem` class to define various types of dynamical models.

BrainPy supports modelings in brain simulation and brain-inspired computing.

All these supports are based on one common concept: **Dynamical System** via `brainpy.DynamicalSystem`.

#### What is `DynamicalSystem`

A `DynamicalSystem` defines the updating rule of the model at single time step.

1. For models with state, `DynamicalSystem` defines the state transition from $t$ to $t + dt$, i.e., $S(t+dt)=F(S(t),x,t,dt)$, where $S$ is the state, $x$ is input, $t$ is the time, and $dt$ is the time step. This is the case for recurrent neural networks (like GRU, LSTM), neuron models (like HH, LIF), or synapse models which are widely used in brain simulation.
2. However, for models in deep learning, like convolution and fully-connected linear layers, `DynamicalSystem` defines the input-to-output mapping, i.e., $y=F(x,t)$.

![img](https://brainpy.readthedocs.io/en/latest/_images/dynamical_system.png)

#### How to define `DynamicalSystem`

```python
class YourDynamicalSystem(bp.DynamicalSystem):
    def update(self, x):
        ...
```

Instead of input x, there are shared arguments across all nodes/layers in the network:

- the current time `t`, or
- the current running index `i`, or
- the current time step `dt`, or
- the current phase of training or testing `fit=True/False`.

Here, it is necessary to explain the usage of `bp.share`.

- `bp.share.save( )`: The function saves shared arguments in the global context. User can save shared arguments in tow ways, for example, if user want to set the current time `t=100`, the current time step `dt=0.1`,the user can use `bp.share.save("t",100,"dt",0.1)` or `bp.share.save(t=100,dt=0.1)`.
- `bp.share.load( )`: The function gets the shared data by the `key`, for example, `bp.share.load("t")`.
- `bp.share.clear_shargs( )`: The function clears the specific shared arguments in the global context, for example, `bp.share.clear_shargs("t")`.
- `bp.share.clear( )`: The function clears all shared arguments in the global context.

#### How to run `DynamicalSystem`

As we have stated above that `DynamicalSystem` only defines the updating rule at single time step, to run a `DynamicalSystem` instance over time, we need a for loop mechanism.

![img](https://brainpy.readthedocs.io/en/latest/_images/dynamical_system_and_dsrunner.png)

##### `brainpy.math.for_loop`

`for_loop` is a structural control flow API which runs a function with the looping over the inputs. Moreover, this API just-in-time compile the looping process into the machine code.

```python
inputs = bp.inputs.section_input([0., 6.0, 0.], [100., 200., 100.])
indices = np.arange(inputs.size)

def run(i, x):
    neu.step_run(i, x)
    return neu.V.value

vs = bm.for_loop(run, (indices, inputs), progress_bar=True)
```

##### `brainpy.LoopOverTime`

Different from `for_loop`, `brainpy.LoopOverTime` is used for constructing a dynamical system that automatically loops the model over time when receiving an input.

`for_loop` runs the model over time. While `brainpy.LoopOverTime` creates a model which will run the model over time when calling it.

```python
net2.reset_state(batch_size=10)
looper = bp.LoopOverTime(net2)
out = looper(currents)
```

##### `brainpy.DSRunner`

**Initializing a `DSRunner`**

Generally, we can initialize a runner for dynamical systems with the format of:

```
runner = DSRunner(target=instance_of_dynamical_system,
                  inputs=inputs_for_target_DynamicalSystem,
                  monitors=interested_variables_to_monitor,
                  dyn_vars=dynamical_changed_variables,
                  jit=enable_jit_or_not,
                  progress_bar=report_the_running_progress,
                  numpy_mon_after_run=transform_into_numpy_ndarray
                  )
```

- `target` specifies the model to be simulated. It must an instance of brainpy.DynamicalSystem.
- `inputs` is used to define the input operations for specific variables.
  - It should be the format of `[(target, value, [type, operation])]`, where `target` is the input target, `value` is the input value, `type` is the input type (such as “fix”, “iter”, “func”), `operation` is the operation for inputs (such as “+”, “-”, “*”, “/”, “=”). Also, if you want to specify multiple inputs, just give multiple `(target, value, [type, operation])`, such as `[(target1, value1), (target2, value2)]`.
  - It can also be a function, which is used to manually specify the inputs for the target variables. This input function should receive one argument `tdi` which contains the shared arguments like time `t`, time step `dt`, and index `i`.
- `monitors` is used to define target variables in the model. During the simulation, the history values of the monitored variables will be recorded. It can also to monitor variables by callable functions and it should be a `dict`. The `key` should be a string for later retrieval by `runner.mon[key]`. The `value` should be a callable function which receives an argument: `tdt`.
- `dyn_vars` is used to specify all the dynamically changed [variables](https://brainpy.readthedocs.io/en/latest/tutorial_math/variables.html) used in the `target` model.
- `jit` determines whether to use JIT compilation during the simulation.
- `progress_bar` determines whether to use progress bar to report the running progress or not.
- `numpy_mon_after_run` determines whether to transform the JAX arrays into numpy ndarray or not when the network finishes running.

**Running a `DSRunner`**

After initialization of the runner, users can call `.run()` function to run the simulation. The format of function `.run()` is showed as follows:

```python
runner.run(duration=simulation_time_length,
           inputs=input_data,
           reset_state=whether_reset_the_model_states,
           shared_args=shared_arguments_across_different_layers,
           progress_bar=report_the_running_progress,
           eval_time=evaluate_the_running_time
           )
```

- `duration` is the simulation time length.
- `inputs` is the input data. If `inputs_are_batching=True`, `inputs` must be a PyTree of data with two dimensions: `(num_sample, num_time, ...)`. Otherwise, the `inputs` should be a PyTree of data with one dimension: `(num_time, ...)`.
- `reset_state` determines whether to reset the model states.
- `shared_args` is shared arguments across different layers. All the layers can access the elements in `shared_args`.
- `progress_bar` determines whether to use progress bar to report the running progress or not.
- `eval_time` determines whether to evaluate the running time.

### Monitors

```python
# initialize monitor through a list of strings
runner1 = bp.DSRunner(target=net,
                      monitors=['E.spike', 'E.V', 'I.spike', 'I.V'],  # 4 elements in monitors
                      inputs=[('E.input', 20.), ('I.input', 20.)],
                      jit=True)
```

Once we call the runner with a given time duration, the monitor will automatically record the variable evolutions in the corresponding models. Afterwards, users can access these variable trajectories by using .mon.[variable_name]. The default history times .mon.ts will also be generated after the model finishes its running. Let’s see an example.

```python
runner1.run(100.)
bp.visualize.raster_plot(runner1.mon.ts, runner1.mon['E.spike'], show=True)
```

**Initialization with index specification**

```python
monitors=[('E.spike', [1, 2, 3]),  # monitor values of Variable at index of [1, 2, 3]
                                'E.V'],  # monitor all values of Variable 'V'

```

> The monitor shape of "E.V" is (run length, variable size) = (1000, 3200)
> The monitor shape of "E.spike" is (run length, index size) = (1000, 3)

**Explicit monitor target**

```python
monitors={'spike': net.E.spike, 'V': net.E.V},
```

> The monitor shape of "V" is = (1000, 3200)
> The monitor shape of "spike" is = (1000, 3200)

**Explicit monitor target with index specification**

```python
monitors={'E.spike': (net.E.spike, [1, 2]),  # monitor values of Variable at index of [1, 2]
                                'E.V': net.E.V},  # monitor all values of Variable 'V'
```

> The monitor shape of "E.V" is = (1000, 3200)
> The monitor shape of "E.spike" is = (1000, 2)

### Inputs

In brain dynamics simulation, various inputs are usually given to different units of the dynamical system. In BrainPy, `inputs` can be specified to runners for dynamical systems. The aim of `inputs` is to mimic the input operations in experiments like Transcranial Magnetic Stimulation (TMS) and patch clamp recording.

`inputs` should have the format like `(target, value, [type, operation])`, where

- `target` is the target variable to inject the input.
- `value` is the input value. It can be a scalar, a tensor, or a iterable object/function.
- `type` is the type of the input value. It support two types of input: `fix` and `iter`. The first one means that the data is static; the second one denotes the data can be iterable, no matter whether the input value is a tensor or a function. The `iter` type must be explicitly stated.
- `operation` is the input operation on the target variable. It should be set as one of `{ + , - , * , / , = }`, and if users do not provide this item explicitly, it will be set to ‘+’ by default, which means that the target variable will be updated as `val = val + input`.

#### Static inputs

```python
runner6 = bp.DSRunner(target=net,
                      monitors=['E.spike'],
                      inputs=[('E.input', 20.), ('I.input', 20.)],  # static inputs
                      jit=True)
runner6.run(100.)
bp.visualize.raster_plot(runner6.mon.ts, runner6.mon['E.spike'])
```

#### Iterable inputs

```python
I, length = bp.inputs.section_input(values=[0, 20., 0],
                                    durations=[100, 1000, 100],
                                    return_length=True,
                                    dt=0.1)

runner7 = bp.DSRunner(target=net,
                      monitors=['E.spike'],
                      inputs=[('E.input', I, 'iter'), ('I.input', I, 'iter')],  # iterable inputs
                      jit=True)
runner7.run(length)
bp.visualize.raster_plot(runner7.mon.ts, runner7.mon['E.spike'])
```

## Run a built-in HH model

[Using Built-in Models — BrainPy documentation](https://brainpy.readthedocs.io/en/latest/tutorial_building/overview_of_dynamic_model.html)

```python
import brainpy as bp
import brainpy.math as bm

current, length = bp.inputs.section_input(values=[0., bm.asarray([1., 2., 4., 8., 10., 15.]), 0.],
                                         durations=[10, 2, 25],
                                         return_length=True)

hh_neurons = bp.neurons.HH(current.shape[1])

runner = bp.DSRunner(hh_neurons, monitors=['V', 'm', 'h', 'n'], inputs=('input', current, 'iter'))

runner.run(length)
```



## Run a HH model from scratch

The mathematic expression of the HH model


$$
\left\{\begin{aligned}&c\frac{\mathrm{d}V}{\mathrm{d}t}=-\bar{g}_\text{Na}m^3h(V-E_\text{Na})-\bar{g}_\text{K}n^4(V-E_\text{K})-\bar{g}_\text{L}(V-E_\text{L})+I_\text{ext},\\&\frac{\mathrm{d}n}{\mathrm{d}t}=\phi\left[\alpha_n(V)(1-n)-\beta_n(V)n\right]\\&\frac{\mathrm{d}m}{\mathrm{d}t}=\phi\left[\alpha_m(V)(1-m)-\beta_m(V)m\right],\\&\frac{\mathrm{d}h}{\mathrm{d}t}=\phi\left[\alpha_h(V)(1-h)-\beta_h(V)h\right],\end{aligned}\right.
$$

$$
\begin{aligned}\alpha_n(V)&=\frac{0.01(V+55)}{1-\exp\left(-\frac{V+55}{10}\right)},\quad\beta_n(V)&=0.125\exp\left(-\frac{V+65}{80}\right),\\\alpha_h(V)&=0.07\exp\left(-\frac{V+65}{20}\right),\quad\beta_n(V)&=\frac{1}{\left(\exp\left(-\frac{V+55}{10}\right)+1\right)},\\\alpha_m(V)&=\frac{0.1(V+40)}{1-\exp\left(-(V+40)/10\right)},\quad\beta_m(V)&=4\exp\left(-(V+65)/18\right).\end{aligned}
$$

$$
\phi=Q_{10}^{(T-T_{\mathrm{base}})/10}
$$

V: the membrane potential

n: activation variable of the Kt channel

m: activation variable of the Nat channel

h; inactivation variable of the Nat channe

### Define HH model `class`

- Inherit `bp.dyn.NeuDyn`

```python
import brainpy as bp
import brainpy.math as bm

class HH(bp.dyn.NeuDyn):
    def __init__(self, size,
                ENa=50., gNa=120.,
                Ek=-77., gK=36.,
                EL=-54.387, gL=0.03,
                V_th=0., C=1.0, T=6.3):
        super(HH, self).__init__(size=size)
```

### Initialization

```python
import brainpy as bp
import brainpy.math as bm

class HH(bp.dyn.NeuDyn):
    def __init__(self, size,
                ENa=50., gNa=120.,
                Ek=-77., gK=36.,
                EL=-54.387, gL=0.03,
                V_th=0., C=1.0, T=6.3):
        super(HH, self).__init__(size=size)
        
        # parameters
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNA = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.V_th = V_th
        self.T_base = 6.3
        self.phi = 3.0 ** ((T - self.T_base) / 10.0)
        
        # variable
        self.V = bm.Variable(-70.68 * bm.ones(self.num))
        self.m = bm.Variable(0.0266 * bm.ones(self.num))
        self.h = bm.Variable(0.772 * bm.ones(self.num))
        self.n = bm.Variable(0.235 * bm.ones(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
        
        # 定义积分函数
    	self.integral = bp.odeint(f=self.derivative, method='exp_auto')
```

### Define the derivative function

```python
@property
def derivative(self):
    return bp.JointEq(self.dV, self.dm, self.dh, self.dn)

def dV(self, V, t, m, h, n, Iext):
    I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
    I_K = (self.gK * n ** 4.0) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / self.C
    return dVdt

def dm(self, m, t, V):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m
    return self.phi * dmdt

def dh(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h
    return self.phi * dhdt

def dn(self, n, t, V):
    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n
    return self.phi * dndt
```

### Complete the `update()` function

```python
def update(self, x=None):
    t = bp.share.load('t')
    dt = bp.share.load('dt')
    # TODO: 更新变量V, m, h, n, 暂存在V, m, h, n中
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, t, self.input, dt=dt)

    #判断是否发生动作电位
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    # 更新最后一次脉冲发放时间
    self.t_last_spike.value = bm.where(self.spike, t, self.t_last_spike)

    # TODO: 更新变量V, m, h, n的值
    self.V.value = V
    self.m.value = m
    self.h.value = h
    self.n.value = n

    #重置输入
    self.input[:] = 0
```

### Simulation

```python
current, length = bp.inputs.section_input(values=[0., bm.asarray([1., 2., 4., 8., 10., 15.]), 0.],
                                          durations=[10, 2, 25],
                                          return_length=True)

hh_neurons = HH(current.shape[1])

runner = bp.DSRunner(hh_neurons, monitors=['V', 'm', 'h', 'n'], inputs=('input', current, 'iter'))

runner.run(length)
```

### Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V (mV)', plot_ids=np.arange(current.shape[1]))

plt.plot(runner.mon.ts, bm.where(current[:, -1]>0, 10, 0) - 90.)

plt.figure()

plt.plot(runner.mon.ts, runner.mon.m[:, -1])
plt.plot(runner.mon.ts, runner.mon.h[:, -1])
plt.plot(runner.mon.ts, runner.mon.n[:, -1])
plt.legend(['m', 'h', 'n'])
plt.xlabel('Time (ms)')
```

## Customize a conductance-based model

电路模拟，写成电导形式

![image-20230824180831033](Notes.assets/image-20230824180831033.png)
$$
\begin{aligned}
\text{gK}& =\bar{g}_\text{K}n^4,  \\
\frac{\mathrm{d}n}{\mathrm{d}t}& =\phi[\alpha_n(V)(1-n)-\beta_n(V)n], 
\end{aligned}
$$
动力学形式描述，引入门框变量$n$
$$
\begin{aligned}
&\alpha_{n}(V) =\frac{0.01(V+55)}{1-\exp(-\frac{V+55}{10})},  \\
&\beta_{n}(V) =0.125\exp\left(-\frac{V+65}{80}\right). 
\end{aligned}
$$
由此式来建模钾离子通道

### Programming an ion channel

#### Three ion channel

```python
import brainpy as bp
import brainpy.math as bm

class IK(bp.dyn.IonChannel):
  def __init__(self, size, E=-77., g_max=36., phi=1., method='exp_auto'):
    super(IK, self).__init__(size)
    self.g_max = g_max
    self.E = E
    self.phi = phi

    self.n = bm.Variable(bm.zeros(size))  # variables should be packed with bm.Variable
    
    self.integral = bp.odeint(self.dn, method=method)

  def dn(self, n, t, V):
    alpha_n = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta_n = 0.125 * bm.exp(-(V + 65) / 80)
    return self.phi * (alpha_n * (1. - n) - beta_n * n)

  def update(self, V):
    t = bp.share.load('t')
    dt = bp.share.load('dt')
    self.n.value = self.integral(self.n, t, V, dt=dt)

  def current(self, V):
    return self.g_max * self.n ** 4 * (self.E - V)
```

```python
class INa(bp.dyn.IonChannel):
  def __init__(self, size, E= 50., g_max=120., phi=1., method='exp_auto'):
    super(INa, self).__init__(size)
    self.g_max = g_max
    self.E = E
    self.phi = phi

    self.m = bm.Variable(bm.zeros(size))  # variables should be packed with bm.Variable
    self.h = bm.Variable(bm.zeros(size))
    
    self.integral_m = bp.odeint(self.dm, method=method)
    self.integral_h = bp.odeint(self.dh, method=method)

  def dm(self, m, t, V):
    # TODO: 计算dm/dt
    alpha_m = 0.11 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta_m = 4 * bm.exp(-(V + 65) / 18)
    return self.phi * (alpha_m * (1. - m) - beta_m * m)

  def dh(self, h, t, V):
    # TODO: 计算dh/dt
    alpha_h = 0.07 * bm.exp(-(V + 65) / 20)
    beta_h = 1. / (1 + bm.exp(-(V + 35) / 10))
    return self.phi * (alpha_h * (1. - h) - beta_h * h)

  def update(self, V):
    t = bp.share.load('t')
    dt = bp.share.load('dt')
    # TODO: 更新self.m, self.h
    self.m.value = self.integral_m(self.m, t, V, dt=dt)
    self.h.value = self.integral_h(self.h, t, V, dt=dt)

  def current(self, V):
    return self.g_max * self.m ** 3 * self.h * (self.E - V)
```

```python
class IL(bp.dyn.IonChannel):
  def __init__(self, size, E=-54.39, g_max=0.03):
    super(IL, self).__init__(size)
    self.g_max = g_max
    self.E = E

  def current(self, V):
    return self.g_max * (self.E - V)
  def update(self, V):
    pass
```

#### Build a HH model with ion channels

**Using customized ion channels**

```python
class HH(bp.dyn.CondNeuGroup):
  def __init__(self, size):
    super(HH, self).__init__(size, V_initializer=bp.init.Uniform(-80, -60.))
    # TODO: 初始化三个离子通道
    self.IK = IK(size, E=-77., g_max=36.)
    self.INa = INa(size, E=50., g_max=120.)
    self.IL = IL(size, E=-54.39, g_max=0.03)
```

**Using built-in ion channels**

```python
class HH(bp.dyn.CondNeuGroup):
    def __init__(self, size):
        super().__init__(size)
        
        self.INa = bp.channels.INa_HH1952(size)
        self.IK = bp.channels.IK_HH1952(size)
        self.IL = bp.cahnnels.IL(size, E=-54.387, g_max=0.03)
```

#### Simulation

```python
neu = HH(1)

runner = bp.DSRunner(
    neu, 
    monitors=['V', 'IK.n', 'INa.m', 'INa.h'], 
    inputs=('input', 1.698)  # near the threshold current
)

runner.run(200)  # the running time is 200 ms

import matplotlib.pyplot as plt

plt.plot(runner.mon['ts'], runner.mon['V'])
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')
plt.savefig("HH.jpg")
plt.show()

plt.figure(figsize=(6, 2))
plt.plot(runner.mon['ts'], runner.mon['IK.n'], label='n')
plt.plot(runner.mon['ts'], runner.mon['INa.m'], label='m')
plt.plot(runner.mon['ts'], runner.mon['INa.h'], label='h')
plt.xlabel('t (ms)')
plt.legend()
plt.savefig("HH_channels.jpg")

plt.show()
```

![image-20230824184016011](Notes.assets/image-20230824184016011.png)
