from matris import Game, GameOver, WIDTH, HEIGHT
import time
import random

class Agent(object):
    
    def __init__(self):
        print("New agent initialized")

    def run_episode(self, draw_screen=False):
        game = Game()
        game.main(draw_screen=draw_screen, comp_control=True)
        while True:
            try:
                rotation, position = self.pick_action(game.matris.current_state())
                if draw_screen:
                    time.sleep(0.2)
                score = game.matris.computer_update(rotation, position)
                # return whether done and next state, maybe can set next_state = current state at the start of the loop if it's set
                if draw_screen: 
                    time.sleep(0.2)
                print(score)
                if draw_screen:
                    game.redraw()
            except GameOver:
                print("episode over")
                if draw_screen:
                    input("press enter to exit")
                return

    def pick_action(self, state):
        print("pick action called")
        return (random.choice(range(4)), random.choice(range(9)))
        


if __name__ == '__main__':
    agent = Agent()
    agent.run_episode(draw_screen=False)