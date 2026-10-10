# Snake Game

## Installation

Install the required dependencies:

>
> `pip3 install -r requirements`

Then just start the program.

> `python3 snake.py`

## Controls

Control the game using the following keys:

- Space - Toggle rendering.
Disabling rendering significantly speeds up training. 

- Delete to kill snake.

Enjoy yourself!

## Config

> The `config.toml` contains the configuration parameters. 

### Parameters

- **`inner_layer`** is the size of hidden layer in the Pytorch model.
- **`trainer_gamma`** is discount factor controlling the importance of future rewards. 
- **`learning_rate`** is a key hyperparameter that controls how quickly a model learns by determining the step size during weight updates.
- **`exploration_games`** is a parameter that controls the number of exploration games.
- **`survival_reward`** is a reward for survival.
- **`death_reward`** is a reward for death.
- **`eating_reward`** is a reward for eating food.
- **`limit_games`** is the maximimum number of games per a training session.
- **`seed`** is random seed value.
