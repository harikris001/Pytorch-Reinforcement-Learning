
![pytorch notion](https://github.com/TH-Activities/saturday-hack-night-template/assets/117498997/b3a31a3d-5852-48ab-86ab-1fe5dd336551)




# Reinforcemnt Learning For Games

## Overview
This repository contains reinforcement learning models designed to master classic games such as Snake, Ping Pong, and Lunar Lander. The models are built using PyTorch and trained in environments created by Gym.
## Team members
1. [Harikrishna R](https://github.com/harikris001)
## Video - product walkthrough



https://github.com/harikris001/Pytorch-Reinforcement-Learning/assets/85405666/9e904a1d-b4ac-47e0-abb9-704d8e8aa91e






## How it Works ?

The project utilizes deep reinforcement learning techniques to train agents that can autonomously play and excel at the mentioned games. Here's a breakdown of how it works:

1.  **Environment Setup**: Each game (Snake, Ping Pong, Lunar Lander) is set up as an environment using Gym.
2.  **Model Building**: Neural networks are built and trained using PyTorch (referred to here as "Fire Torch" due to its dynamic and fast computations).
3.  **Training Phase**: The agents learn through episodes of playing, where they update their strategy based on the rewards received.
4.  **Model Saving**: After a set number of iterations, the trained models are saved for later use or further training.
## Libraries used
Library Name :
- [PyTorch](https://pytorch.org/docs/stable/index.html) 2.2.2
- [Gym](https://openai.com/research/openai-gym-beta) 0.26.2
- [pygame](https://www.pygame.org/docs/)

## How to configure
To get started with the project, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/yourgithubprofile/Reinforcement-Learning-Arcade.git
    ```
2. Install the required libraries and dependencies:
   ```
   pip install -r requirements.txt
   ```

   *Note: You need CUDA ToolKit installed if you require faster pocessing without cuda toolkit pip install requirments may give error. if so, ignore error and just type pip install torch in terminal*
## How to Run
1. Navigate to the project directory:
   ```
   cd snake-ai
   ```
   ```
   cd pong-atri
   ```
   ```
   cd lunar-lander
   ```
2. Run the model for a specific game. For example, to run the Snake game model:
   ```
   python agent.py 
   ```
   for snake ai
   ```
   python run.py
   ```
   for pong-atri
   ```
   python runner.py
   ```
   for lunar-lander
