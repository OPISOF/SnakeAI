import os
import random

import matplotlib.pyplot as plt
import pygame
import torch
from torch import nn, optim

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


class SnakeGame:
    def __init__(self):
        self.count = 0
        self.reset()

    def reset(self):
        self.snake = [[100, 100], [100 + BLOCK, 100]]
        self.direction = "DOWN"
        self.food = spawn_food(self.snake)
        self.score = 0
        self.count += 1

    def step(self, action):
        reward = SURVIVAL_REWARD

        self.decode_turn(action)
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
        danger.append(is_crash(danger_straight[0], danger_straight[1], self.snake))
        danger.append(is_crash(danger_right[0], danger_right[1], self.snake))

        general_state = []

        general_state.append(head_x > self.food[0])
        general_state.append(head_x < self.food[0])
        general_state.append(head_y > self.food[1])
        general_state.append(head_y < self.food[1])

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

    def decode_turn(self, action):
        index = COMPASS.index(self.direction)

        if action[1] == 1:
            pass
        elif action[0] == 1:
            self.direction = COMPASS[index - 1]
        elif action[2] == 1:
            self.direction = COMPASS[(index + 1) % len(COMPASS)]


class Renderer:
    def __init__(self):
        pygame.init()
        self.enabled = True
        self.snake_colour = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def draw_frame(self, game):
        self.display.fill("white")

        for x_position, y_position in game.snake:
            pygame.draw.rect(
                self.display,
                self.snake_colour,
                pygame.Rect(x_position, y_position, BLOCK, BLOCK),
                width=4,
            )

        pygame.draw.rect(
            self.display, "black", pygame.Rect(game.food[0], game.food[1], BLOCK, BLOCK)
        )

        score_text = self.font.render(f"Score: {game.score}", True, "black")
        self.display.blit(score_text, (25, 25))

        pygame.display.flip()
        self.clock.tick(FPS)

    def switch_render(self):
        self.enabled = not self.enabled


class Agent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(7, INNER_LAYER),
            nn.ReLU(),
            nn.Linear(INNER_LAYER, 3),
        )
        self.optimizer = optim.Adam(self.model.parameters(), LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def choose_action(self, state, count):
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)

        prediction = self.model(state0)
        move = torch.argmax(prediction).item()

        if EXPLORATION_GAMES - count > random.randint(0, EXPLORATION_GAMES):
            move = random.randint(0, 2)

        final_move[move] = 1
        return final_move

    def agent_train(self, state, action, next_state, reward, is_game_over):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        is_game_over = torch.tensor(is_game_over, dtype=torch.float).unsqueeze(0)

        pred = self.model(state)
        target = pred.clone()

        q_new = reward[0]
        if not is_game_over[0]:
            q_new = reward[0] + TRAINER_GAMMA * torch.max(self.model(next_state))

        target[0][torch.argmax(action).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Plot:
    def __init__(self):
        plt.ion()
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
        top_score = max(
            max(self.mean_score_series, default=0), max(self.score_series, default=0)
        )
        plt.ylim(0, top_score + 5)
        plt.draw()
        plt.pause(0.001)

    def refresh_score(self, score):
        self.score_series.append(score)
        self.mean_score_series.append(sum(self.score_series) / len(self.score_series))

    def save(self):
        self.update()

        if not os.path.exists("./runs"):
            os.makedirs("./runs")

        number = len(os.listdir("./runs"))
        plt.savefig(f"./runs/run{number}")


def is_crash(x, y, snake):
    if x >= SCREEN_WIDTH or y >= SCREEN_HEIGHT or x < 0 or y < 0:
        return True

    if [x, y] in snake[1:]:
        return True
    else:
        return False


def spawn_food(snake):
    while True:
        food = [
            random.randint(0, SCREEN_WIDTH // BLOCK - 1) * BLOCK,
            random.randint(0, SCREEN_HEIGHT // BLOCK - 1) * BLOCK,
        ]
        if food not in snake:
            break

    return food


def main():
    limit_games = int(input("Set game limitation: "))
    game = SnakeGame()
    renderer = Renderer()
    agent = Agent()
    plot = Plot()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                plot.save()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    renderer.switch_render()

                if event.key == pygame.K_TAB:
                    renderer.switch_render()

                if event.key == pygame.K_BACKSPACE:
                    game.reset()

        state = game.get_state()
        action = agent.choose_action(state, game.count)

        is_game_over, reward = game.step(action)

        if renderer.enabled:
            renderer.draw_frame(game)

        next_state = game.get_state()

        agent.agent_train(state, action, next_state, reward, is_game_over)

        if is_game_over:
            plot.refresh_score(game.score)

            if game.count >= limit_games:
                plot.save()
                return

            if game.count % PLOT_RATE == 0:
                plot.update()

            game.reset()


if __name__ == "__main__":
    main()
