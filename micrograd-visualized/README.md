# Micrograd Visualization System

## Overview
[Micrograd](https://github.com/karpathy/micrograd) is a minimalistic, elegant autograd engine and neural network library created by Andrej Karpathy. It serves as an excellent resource for learning the foundational elements of deep learning because it implements everything from the ground up without relying on complex, heavy frameworks. 

This educational value is unmatched, but visualizing what happens under the hood during the training process can be challenging. This project adds a comprehensive visualization layer on top of Micrograd, providing interactive tools to make neural network behavior, training dynamics, and computation graphs much easier to understand.

## Features
This repository extends Micrograd with the following key capabilities:
- **Neural Network Architecture Visualization:** A high-level view of the network layers and connections.
- **Neuron Computation Graph Inspection:** An interactive way to drill down into individual neurons to see their weights, biases, inputs, and gradients.
- **Training Loss Curve Visualization:** Live-updating charts to track the progression of the loss function during training.
- **Dataset Sampling Heatmap:** A visual representation indicating which parts of the dataset the model is currently feeding on.
- **Interactive Training Timeline:** A step-by-step scrubber that allows you to replay the training process and observe how parameters evolve over time.

## Why This Project Exists
This project was primarily created as a learning exercise to gain a deeper, more intuitive understanding of the core concepts of machine learning, including:
- Automatic differentiation
- Backpropagation
- Neural network parameter updates

Building visualization tools directly on top of the computation graph proved to be an incredibly effective way to see exactly how gradients propagate backwards through the network and how individual weights are adjusted. 

## Acknowledgements
A huge thank you to **Andrej Karpathy** and the original [Micrograd](https://github.com/karpathy/micrograd) library. Micrograd stands as an exceptional educational resource for understanding neural networks from absolute first principles. This project is entirely built upon the beautiful simplicity of his work.

## Development Experience
The visualization layer, debugging interface, and integration process for this project were implemented while experimenting with the **Antigravity IDE**. Using an agentic, AI-powered coding assistant significantly accelerated the development and experimentation cycle. Overall, building this visualization layer was a highly rewarding and hands-on learning experience.

## How to Run the Demo
To see the visualization system in action, follow these steps:
1. **Install dependencies:** Ensure all required packages are installed in your Python environment.
2. **Run Jupyter Notebook:** Launch a local Jupyter Notebook server.
3. **Open the Demo:** Navigate to and open `demo.ipynb`.
4. **Run the Notebook:** Execute all the cells in the notebook, including the visualization sections, to train the model and launch the interactive debugger.

## Dependencies
Ensure you have the following libraries installed:
- `graphviz`
- `matplotlib`
- `ipywidgets`
- `numpy`

## Future Improvements
There are several exciting directions for expanding this visualization system in the future:
- **Gradient flow visualization across layers:** Global mapping of how gradients decay or explode across the entire network architecture.
- **Parameter evolution tracking:** Direct visual tracking of how specific weight matrices shift over an entire training cycle.
- **Improved interactive dashboards:** More fluid and browser-friendly UI elements to simplify navigation.
