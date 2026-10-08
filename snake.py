import random
import sys

import matplotlib.pyplot as plt
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


SCREEN_WIDTH, SCREEN_HEIGHT = 720, 480
BLOCK = 20
FPS = 100
PLOT_RATE = 100


INNER_LAYER = 512
TRAINER_GAMMA = 0.89
LEARNING_RATE = 0.0025
EXPLORATION_GAMES = 1000
SURVIVAL_REWARD = 0
DEATH_REWARD = -10
EATING_REWARD = 10


COMPASS = ["UP", "RIGHT", "DOWN", "LEFT"]


SNAKE_COLOUR = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
)


class SnakeGame:
    def __init__(self):
        self.game_count = 0
        self.reset()

    def reset(self):
        self.snake = [[100, 100], [100 + BLOCK, 100]]
        self.direction = "DOWN"
        self.food = spawn_food(self.snake)
        self.score = 0
        self.game_count+=1

    def game_step(self):
        reward = SURVIVAL_REWARD
        
        head_x, head_y = self.snake[0]

        if self.direction == "DOWN":
            head_y += BLOCK
        elif self.direction == "UP":
            head_y -= BLOCK
        elif self.direction == "RIGHT":
            head_x += BLOCK
        elif self.direction == "LEFT":
            head_x -= BLOCK
        
        if is_crash(head_x, head_y, self.snake):
            reward = DEATH_REWARD
            return True, reward

        self.snake.insert(0, [head_x, head_y])

        if self.snake[0] != self.food:
            self.snake.pop()
        else:
            self.score += 1
            reward = EATING_REWARD
            self.food = spawn_food(self.snake)

        return False, reward
 
    def get_state(self):
        head_x, head_y = self.snake[0]

        if self.direction == "UP":
            danger_left = [head_x - BLOCK, head_y]
            danger_straight = [head_x, head_y - BLOCK]
            danger_right = [head_x + BLOCK, head_y]

        if self.direction == "DOWN":
            danger_left = [head_x + BLOCK, head_y]
            danger_straight = [head_x, head_y + BLOCK]
            danger_right = [head_x - BLOCK, head_y]

        if self.direction == "RIGHT":
            danger_left = [head_x, head_y - BLOCK]
            danger_straight = [head_x + BLOCK, head_y]
            danger_right = [head_x, head_y + BLOCK]

        if self.direction == "LEFT":
            danger_left = [head_x, head_y + BLOCK]
            danger_straight = [head_x - BLOCK, head_y]
            danger_right = [head_x, head_y - BLOCK]

        danger = []

        danger.append(is_crash(danger_left[0], danger_left[1], self.snake))
        danger.append(is_crash(
            danger_straight[0], danger_straight[1], self.snake)
        )
        danger.append(is_crash(
            danger_right[0], danger_right[1], self.snake)
        )

        general_state = []

        general_state.append((head_x > self.food[0]))
        general_state.append((head_x < self.food[0]))
        general_state.append((head_y > self.food[1]))
        general_state.append((head_y < self.food[1]))

        if self.direction == "RIGHT":
            general_state = [
                general_state[2],
                general_state[3],
                general_state[1],
                general_state[0],
            ]
        if self.direction == "DOWN":
            general_state = [
                general_state[1],
                general_state[0],
                general_state[3],
                general_state[2],
            ]
        if self.direction == "LEFT":
            general_state = [
                general_state[3],
                general_state[2],
                general_state[0],
                general_state[1],
            ]

        state = []
        state = danger + general_state

        return state       


class Render:
    def __init__(self):
        self.render = True
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def draw_frame(self, SnakeGame):
        self.display.fill("white")

        for x_position, y_position in SnakeGame.snake:
            pygame.draw.rect(
                self.display,
                SNAKE_COLOUR,
                pygame.Rect(x_position, y_position, BLOCK, BLOCK),
                width=4,
            )

        pygame.draw.rect(
                self.display, 'black', pygame.Rect(
                    SnakeGame.food[0], SnakeGame.food[1], BLOCK, BLOCK
                )
        )

        score_text = self.font.render(f"Score: {SnakeGame.score}", True, "black")
        self.display.blit(score_text, (25, 25))

        pygame.display.flip()
        self.clock.tick(FPS)


class Agent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(7, INNER_LAYER),
            nn.ReLU(),
            nn.Linear(INNER_LAYER, 3),
        )
        self.optimizer = optim.Adam(self.model.parameters(), LEARNING_RATE)
        self.criterion = nn.MSELoss()


    def choose_action(self, agent_state, game_count):
        final_move = [0, 0, 0]
        state0 = torch.tensor(agent_state, dtype=torch.float)

        prediction = self.model(state0)
        move = torch.argmax(prediction).item()

        if EXPLORATION_GAMES - game_count > random.randint(0, EXPLORATION_GAMES):
            move = random.randint(0, 2)

        final_move[move] = 1
        return final_move

    def decode_turn(self, direction, action):
        index = COMPASS.index(direction)

        if action[1] == 1:
            return direction
        if action[0] == 1:
            return COMPASS[index - 1]
        if action[2] == 1:
            return COMPASS[(index + 1) % len(COMPASS)]


    def agent_train(
            self, agent_state, action, agent_new_state,
            reward, is_game_over
    ):
        agent_state = torch.tensor(agent_state, dtype=torch.float).unsqueeze(0)
        agent_new_state = torch.tensor(
                agent_new_state, dtype=torch.float
        ).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        is_game_over = torch.tensor(is_game_over, dtype=torch.float).unsqueeze(0)

        pred = self.model(agent_state)
        target = pred.clone()

        q_new = reward[0]
        if not is_game_over[0]:
            q_new = reward[0] + TRAINER_GAMMA * torch.max(
                    self.model(agent_new_state)
            )

        target[0][torch.argmax(action).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Plot:
    def __init__(self):
        self.score_series = []
        self.mean_score_series = []

    def update(self):
        plt.clf()
        plt.plot(self.score_series, marker="", linestyle="-", color="b")
        plt.plot(self.mean_score_series, marker="", linestyle="-", color="r")
        plt.title("learning...")
        plt.xlabel("Game number")
        plt.ylabel("Score")
        plt.grid(False)
        top_score = max(max(self.mean_score_series, default=0),
                         max(self.score_series, default=0)
        ) 
        plt.ylim(0, top_score + 5)
        plt.draw()
        plt.pause(0.001)


def is_crash(x, y, snake):
    if x >= SCREEN_WIDTH or y >= SCREEN_HEIGHT or x < 0 or y < 0:
        return 1

    if [x, y] in snake[1:]:
        return 1
    else:
        return 0


def spawn_food(snake):
    while True:
        food = [
            random.randint(0, SCREEN_WIDTH // BLOCK - 1) * BLOCK,
            random.randint(0, SCREEN_HEIGHT // BLOCK - 1) * BLOCK,
        ]
        if food not in snake:
            break

    return food



render = True

def main():
    pygame.init()
    plt.ion()
    
    game = SnakeGame()
    renderer = Render()
    agent = Agent()
    plot = Plot()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    renderer.render = False

                if event.key == pygame.K_TAB:
                    renderer.render = True

                if event.key == pygame.K_BACKSPACE:
                    game.reset() 

        agent_state = game.get_state()
        action = agent.choose_action(agent_state, game.game_count)

        game.direction = agent.decode_turn(game.direction, action)

        is_game_over, reward = game.game_step()

        if renderer.render:
            renderer.draw_frame(game)

        agent_new_state = game.get_state()

        agent.agent_train(agent_state, action, agent_new_state,
                          reward, is_game_over
        )

        if is_game_over:
            plot.score_series.append(game.score)

            plot.mean_score_series.append(
                    sum(plot.score_series) / len(plot.score_series)
            )

            if game.game_count % PLOT_RATE == 0:
                plot.update()

            game.reset()


if __name__ == '__main__':
    main()
