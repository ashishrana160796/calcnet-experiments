# Repository Description

This repository contains experimental setup for experiments conducted in 'Exploring numerical exploration with CalcNet' paper
that was published in ICTAI 2019. It was a submission under short paper categories involving new research ideas and concepts
that are under development.  

A small introduction to motivation and whole concept behind this paper is highlighted in this [introductory presentation](CalcNet-ICTAI-2019.pdf).  

# Motivation and Objective of Research
 
The motivation behind this research is developing neural network architecture pipeline that learns fundamentally
"how to compute a numerical expression ?" and breaks downs complex functions like factorial, exponential etc. which the 
current existing architecture are unable to solve.
This network holds the capability to predict for unseen higher range data which highlights that our neural network pipeline
did fundamentally understood the numerical evaluation.
The basic idea is to input an equation, parse it and then use neural network modules evaluate
the expression under specific degree of certainity. Objective of this research is to being able to solve
approximation problems, stochastic quantum mechanics equations and capture hidden nuances in such equations which might be
harder for current mathematical models to formulate.
Like capturing wave-particle duality nuances in _Heisenberg's uncertainty equation_ with our proposed pipeline from the experiment data.
Hence, introduce neural networks as a assisting mean for equations that already involve uncertainity in them and certain hidden trends that are not captured by existing conventional mathematical models.  

Also, we intend to create fusion of conventional algorithms with neural network modules and form a standard functioning pipeline.
It is an unexplored space for creating new neural network architectures with rigity in the form with conventional data structures for support.
Finally, leading to creation of effective neural networks that truly understands functioning of numerical manipulation with
grammer parsing via infix expression evaluation and NALU/NAC module for evaluating basic operators.

# Outcomes from the paper

* CalcNet architecture explained and it's utility explained on small scale experiments.
* New novel approach highlighted to carry out numerical computations with neural networks.
* Resuable code modules and saved models to get you up and running in no time with CalcNet.

# Experiment Content

This repository contains following directories having information and experiments conducted in the abovementioned paper.
We have shown the utilization of the algorithm ,an error propagation methodology and visualizations of NAC/NALU activation
affline layers. Please, refer to the directories below that interests you.

* __calcnet-experiments:__ An equation evaluation related to predicting results of experiments mentioned in paper.
* __calcnet-paper:__ This directory consist of all the assets and research paper content reproducability related files.
* __nalu-plots:__ 3-D Plot diagrams of NAC/NALU activations and its golden ratio based variants.
* __nalu-variants-experiments:__ This contains experimental comparison of standard NAC/NALU variants on multiple operators and functions with their designed golden ratio based variants.

