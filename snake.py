import random

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
SURVIVAL_REWARD = 0.01
DEATH_REWARD = -10
EATING_REWARD = 10


COMPASS = ["UP", "RIGHT", "DOWN", "LEFT"]


SNAKE_COLOUR = (
        random.randint(0, 255), 
        random.randint(0, 255), 
        random.randint(0, 255)
)


def reset(snake, direction, food, score, head_y, head_x, game_count):
    game_count += 1
    score = 0
    snake = [[100, 100], [100 + BLOCK, 100]]
    food = spawn_food(snake)
    direction = "DOWN"
    head_x, head_y = snake[0]

    return snake, direction, food, score, head_y, head_x, game_count


def game_step(food, score, direction, head_x, head_y):
    reward = SURVIVAL_REWARD

    if direction == "DOWN":
        head_y = head_y + BLOCK
    elif direction == "UP":
        head_y = head_y - BLOCK
    elif direction == "RIGHT":
        head_x = head_x + BLOCK
    elif direction == "LEFT":
        head_x = head_x - BLOCK

    if (
        head_x >= SCREEN_WIDTH
        or head_y >= SCREEN_HEIGHT
        or head_x < 0
        or head_y < 0
        or [head_x, head_y] in snake[1:]
    ):
        reward = DEATH_REWARD
        return True, reward, food, score, direction, head_x, head_y

    snake.insert(0, [head_x, head_y])

    if snake[0] != food:
        snake.pop()
    else:
        score += 1
        reward = EATING_REWARD
        food = spawn_food(snake)

    return False, reward, food, score, direction, head_x, head_y


def draw_frame(snake, food):
    display.fill("white")

    for x_position, y_position in snake:
        pygame.draw.rect(
            display,
            SNAKE_COLOUR,
            pygame.Rect(x_position, y_position, BLOCK, BLOCK),
            width=4,
        )

    pygame.draw.rect(
            display, 'black', pygame.Rect(food[0], food[1], BLOCK, BLOCK)
    )

    score_text = font.render(f"Score: {score}", True, "black")
    display.blit(score_text, (25, 25))

    pygame.display.flip()
    clock.tick(FPS)


def check_crash(x, y, snake):
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


def choose_action(agent_state, EXPLORATION_GAMES, game_count):
    final_move = [0, 0, 0]
    state0 = torch.tensor(agent_state, dtype=torch.float)

    prediction = model(state0)
    move = torch.argmax(prediction).item()

    if EXPLORATION_GAMES - game_count > random.randint(0, EXPLORATION_GAMES):
        move = random.randint(0, 2)

    final_move[move] = 1
    return final_move


def decode_turn(direction, action):
    index = COMPASS.index(direction)

    if action[1] == 1:
        return direction
    if action[0] == 1:
        return COMPASS[index - 1]
    if action[2] == 1:
        if index + 1 == 4:
            return COMPASS[0]
        else:
            return COMPASS[index + 1]


def get_state(head_x, head_y, food, snake):
    if direction == "UP":
        danger_left = [head_x - BLOCK, head_y]
        danger_straight = [head_x, head_y - BLOCK]
        danger_right = [head_x + BLOCK, head_y]

    if direction == "DOWN":
        danger_left = [head_x + BLOCK, head_y]
        danger_straight = [head_x, head_y + BLOCK]
        danger_right = [head_x - BLOCK, head_y]

    if direction == "RIGHT":
        danger_left = [head_x, head_y - BLOCK]
        danger_straight = [head_x + BLOCK, head_y]
        danger_right = [head_x, head_y + BLOCK]

    if direction == "LEFT":
        danger_left = [head_x, head_y + BLOCK]
        danger_straight = [head_x - BLOCK, head_y]
        danger_right = [head_x, head_y - BLOCK]

    danger = []

    danger.append(check_crash(danger_left[0], danger_left[1], snake))
    danger.append(check_crash(danger_straight[0], danger_straight[1], snake))
    danger.append(check_crash(danger_right[0], danger_right[1], snake))

    general_state = []

    general_state.append(int(head_x > food[0]))
    general_state.append(int(head_x < food[0]))
    general_state.append(int(head_y > food[1]))
    general_state.append(int(head_y < food[1]))

    if direction == "RIGHT":
        general_state = [
            general_state[2],
            general_state[3],
            general_state[1],
            general_state[0],
        ]
    if direction == "DOWN":
        general_state = [
            general_state[1],
            general_state[0],
            general_state[3],
            general_state[2],
        ]
    if direction == "LEFT":
        general_state = [
            general_state[3],
            general_state[2],
            general_state[0],
            general_state[1],
        ]

    state = []
    state = danger + general_state

    return state


def agent_train(agent_state, action, agent_new_state, reward, gameover):
    agent_state = torch.tensor(agent_state, dtype=torch.float).unsqueeze(0)
    agent_new_state = torch.tensor(agent_new_state, dtype=torch.float).unsqueeze(0)
    action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
    reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
    gameover = torch.tensor(gameover, dtype=torch.float).unsqueeze(0)

    pred = model(agent_state)
    target = pred.clone()

    Q_new = reward[0]
    if not gameover[0]:
        Q_new = reward[0] + TRAINER_GAMMA * torch.max(model(agent_new_state))

    target[0][torch.argmax(action).item()] = Q_new

    optimizer.zero_grad()
    loss = criterion(target, pred)
    loss.backward()
    optimizer.step()


# ===================================


plt.ion()


def plot(mean_score_series, score_series):
    plt.clf()
    plt.plot(score_series, marker="", linestyle="-", color="b")
    plt.plot(mean_score_series, marker="", linestyle="-", color="r")
    plt.title("learning...")
    plt.xlabel("Game number")
    plt.ylabel("Score")
    plt.grid(False)
    plt.ylim(0, max(max(mean_score_series, default=0), max(score_series, default=0)) + 5)
    plt.draw()
    plt.pause(0.001)


# ===================================


pygame.init()


display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)


game_count = 0
score_series = []
mean_score_series = []


model = nn.Sequential(
    nn.Linear(7, INNER_LAYER),
    nn.ReLU(),
    nn.Linear(INNER_LAYER, 3),
)
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
criterion = nn.MSELoss()


snake, direction, food, score, head_y, head_x, game_count = reset(
    [], "DOWN", [], 0, 0, 0, 0
)


render = True


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                render = False

            if event.key == pygame.K_TAB:
                render = True


            if event.key == pygame.K_BACKSPACE:
                snake, direction, food, score, head_y, head_x, game_count = reset(
                    snake, direction, food, score, head_y, head_x, game_count
                )

    agent_state = get_state(head_x, head_y, food, snake)
    action = choose_action(agent_state, EXPLORATION_GAMES, game_count)    

    direction = decode_turn(direction, action)

    gameover, reward, food, score, direction, head_x, head_y = game_step(
        food, score, direction, head_x, head_y
    )

    if render == True:
        draw_frame(snake, food)

    agent_new_state = get_state(head_x, head_y, food, snake)

    agent_train(agent_state, action, agent_new_state, reward, gameover)

    if gameover:
        score_series.append(score)

        total = 0

        for i in score_series:
            total += i

        mean_score_series.append(total / len(score_series))

        if game_count % PLOT_RATE == 0:
            plot(mean_score_series, score_series)

        snake, direction, food, score, head_y, head_x, game_count = reset(
            snake, direction, food, score, head_y, head_x, game_count
        )
