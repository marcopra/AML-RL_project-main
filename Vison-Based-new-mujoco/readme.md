# Starting code for course project of Advanced Machine Learning (AML) 2022
"Sim-to-Real transfer of Reinforcement Learning policies in robotics" exam project.


## Getting started

You can play around with the code on your local machine, and use Google Colab for training on GPUs. When dealing with simple multi-layer perceptrons (MLPs), you can even attempt training on your local machine.

Before starting to implement your own code, make sure to:
1. read and study the material provided
2. read the documentation of the main packages you will be using ([mujoco](https://github.com/deepmind/mujoco), [gymnasium](https://gymnasium.farama.org/), [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html))
3. play around with the code in the template to familiarize with all the tools.

### 1. Local

You can work on your local machine directly, at least for the first stages of the project. By doing so, you will also be able to render the Mujoco environments and visualize what's happening. This code has been tested on Linux with python 3.7, but [mujoco](https://github.com/deepmind/mujoco) provides prebuilt binaries for Windows and MacOS as well.

On Linux, get started on your local machine by executing `pip install -r requirements.txt`. See the official [installation section](https://github.com/deepmind/mujoco#installation) for other operating systems.


### 2. Google Colab

You can also run the code on [Google Colab](https://colab.research.google.com/)

- Download all files contained in the `colab_template` folder in this repo.
- Load the `test_random_policy.ipynb` file on [https://colab.research.google.com/](colab) and follow the instructions on it

NOTE 1: rendering is currently **not** officially supported on Colab, making it hard to see the simulator in action. We recommend that each group manages to play around with the visual interface of the simulator at least once, to best understand what is going on with the Hopper environment.

NOTE 2: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.