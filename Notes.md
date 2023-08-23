[TOC]

# 神经计算建模简介

## 计算神经科学的背景与使命

计算神经科学是**脑科学**对**类脑智能**的**桥梁**

### 两大目标

- 用计算建模的方法来阐明大脑功能的计算原理
- 发展类闹智能的模型和算法

### Prehistory

- 1907 LIF model 
  神经计算的本质
- 1950s HH model 
  电位定量化模型 最fundamental的
- 1960s Roll's cable equation 
  描述信号在轴突和树突怎么传递
- 1970s Amari, Wilson, Cowan et al.
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