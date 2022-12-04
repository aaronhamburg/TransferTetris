#!/usr/bin/env python
import pygame
from pygame import Rect, Surface
import random
import numpy as np

from tetrominoes import Tetromino, list_of_tetrominoes, rotate, piece_range

class GameOver(Exception):
    """Exception used for its control flow properties"""

BGCOLOR = (15, 15, 20)
BORDERCOLOR = (140, 140, 140)

BLOCKSIZE = 30
BORDERWIDTH = 10

MATRIS_OFFSET = 20

MATRIX_WIDTH = 10
MATRIX_HEIGHT = 22

LEFT_MARGIN = 340

WIDTH = MATRIX_WIDTH*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2 + LEFT_MARGIN
HEIGHT = (MATRIX_HEIGHT-2)*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2

TRICKY_CENTERX = WIDTH-(WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2))/2

VISIBLE_MATRIX_HEIGHT = MATRIX_HEIGHT - 2

pygame.init()

class Matris(object):
    def __init__(self, screen: Surface, draw_screen=True):
        
        self.draw_screen = draw_screen

        if draw_screen:
            self.surface = screen.subsurface(Rect((MATRIS_OFFSET+BORDERWIDTH, MATRIS_OFFSET+BORDERWIDTH),
                                                (MATRIX_WIDTH * BLOCKSIZE, (MATRIX_HEIGHT-2) * BLOCKSIZE)))

        self.matrix = dict()
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                self.matrix[(y,x)] = None
        """
        `self.matrix` is the current state of the tetris board, that is, it records which squares are
        currently occupied. It does not include the falling tetromino. The information relating to the
        falling tetromino is managed by `self.set_tetrominoes` instead. When the falling tetromino "dies",
        it will be placed in `self.matrix`.
        """

        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.set_tetrominoes()
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.4 # Move down every 400 ms

        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.level = 1
        self.score = 0
        self.lines = 0

        self.combo = 1 # Combo will increase when you clear lines with several tetrominos in a row
        
        self.paused = False

    # returns a 2D np array representing the board state and current and next pieces
    def current_state(self):
        board_state = np.empty((MATRIX_HEIGHT, MATRIX_WIDTH))
        for key in self.matrix.keys():
            val = self.matrix.get(key)
            if val == None or val == False:
                board_state[(key[0], key[1])] = 0
            elif val[0] == 'block':
                board_state[(key[0], key[1])] = 1
            else:
                board_state[(key[0], key[1])] = 0

        piece_state = np.zeros((4, MATRIX_WIDTH))
        cur_piece = self.piece_to_array(self.current_tetromino)
        next_piece = self.piece_to_array(self.next_tetromino)
        cur_piece_size = cur_piece.shape[0]
        next_piece_size = next_piece.shape[0]
        piece_state[0:int(cur_piece_size), 0:int(cur_piece_size)] = cur_piece
        piece_state[0:int(next_piece_size), 5:int(5+next_piece_size)] = next_piece

        return np.concatenate((piece_state, board_state), axis=0)

    # converts a Tetromino to a 2D np array containing 1's and 0's
    def piece_to_array(self, piece: Tetromino):
        piece_shape = np.asarray(piece.shape)
        for col in range(len(piece_shape)):
            for row in range(len(piece_shape)):
                piece_shape[col, row] = 1 if piece_shape[col, row] else 0
        return piece_shape


    def set_tetrominoes(self):
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)

    
    def hard_drop(self):
        amount = 0
        while self.request_movement('down'):
            amount += 1
        self.score += 10*amount

        self.lock_tetromino()

    def computer_update(self, rotation, position):
        # TODO: check that rotation and position are valid
        # *may* not need to handle this since the environment will prevent illegal moves

        start_score = self.score

        for i in range(rotation):
            self.request_rotation()
        
        left, right = piece_range(self.rotated(self.tetromino_rotation))
        if position not in range(-left, 10 - right):
            print("invalid position :(")
        

        start_pos = self.tetromino_position[1]
        if position <= start_pos:
            for i in range(start_pos - position):
                self.request_movement('left')
        else:
            for i in range(position - start_pos):
                self.request_movement('right')
        
        self.hard_drop()

        return self.score - start_score
        

    def update(self, timepassed):
        self.needs_redraw = False
        
        pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
        unpressed = lambda key: event.type == pygame.KEYUP and event.key == key

        events = pygame.event.get()
        
        for event in events:
            if pressed(pygame.K_p):
                self.surface.fill((0,0,0))
                self.needs_redraw = True
                self.paused = not self.paused
            elif event.type == pygame.QUIT:
                self.gameover(full_exit=True)
            elif pressed(pygame.K_ESCAPE):
                self.gameover()

        if self.paused:
            return self.needs_redraw

        for event in events:
            if pressed(pygame.K_SPACE):
                self.hard_drop()
            elif pressed(pygame.K_UP) or pressed(pygame.K_w):
                self.request_rotation()

            elif pressed(pygame.K_LEFT) or pressed(pygame.K_a):
                self.request_movement('left')
                self.movement_keys['left'] = 1
            elif pressed(pygame.K_RIGHT) or pressed(pygame.K_d):
                self.request_movement('right')
                self.movement_keys['right'] = 1

            elif unpressed(pygame.K_LEFT) or unpressed(pygame.K_a):
                self.movement_keys['left'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2
            elif unpressed(pygame.K_RIGHT) or unpressed(pygame.K_d):
                self.movement_keys['right'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2




        self.downwards_speed = self.base_downwards_speed ** (1 + self.level/10.)

        self.downwards_timer += timepassed
        downwards_speed = self.downwards_speed*0.10 if any([pygame.key.get_pressed()[pygame.K_DOWN],
                                                            pygame.key.get_pressed()[pygame.K_s]]) else self.downwards_speed
        if self.downwards_timer > downwards_speed:
            if not self.request_movement('down'):
                self.lock_tetromino()

            self.downwards_timer %= downwards_speed


        if any(self.movement_keys.values()):
            self.movement_keys_timer += timepassed
        if self.movement_keys_timer > self.movement_keys_speed:
            self.request_movement('right' if self.movement_keys['right'] else 'left')
            self.movement_keys_timer %= self.movement_keys_speed
        
        return self.needs_redraw

    def draw_surface(self):
        with_tetromino = self.blend(matrix=self.place_shadow())

        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):

                #                                       I hide the 2 first rows by drawing them outside of the surface
                block_location = Rect(x*BLOCKSIZE, (y*BLOCKSIZE - 2*BLOCKSIZE), BLOCKSIZE, BLOCKSIZE)
                if with_tetromino[(y,x)] is None:
                    self.surface.fill(BGCOLOR, block_location)
                else:
                    if with_tetromino[(y,x)][0] == 'shadow':
                        self.surface.fill(BGCOLOR, block_location)
                    
                    self.surface.blit(with_tetromino[(y,x)][1], block_location)
                    
    def gameover(self, full_exit=False):
        """
        Gameover occurs when a new tetromino does not fit after the old one has died, either
        after a "natural" drop or a hard drop by the player. That is why `self.lock_tetromino`
        is responsible for checking if it's game over.
        """
        
        if full_exit:
            exit()
        else:
            raise GameOver("Sucker!")

    def place_shadow(self):
        posY, posX = self.tetromino_position
        while self.blend(position=(posY, posX)):
            posY += 1

        position = (posY-1, posX)

        return self.blend(position=position, shadow=True)

    def fits_in_matrix(self, shape, position):
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if self.matrix.get((y, x), False) is False and shape[y-posY][x-posX]: # outside matrix
                    return False

        return position
                    

    def request_rotation(self):
        rotation = (self.tetromino_rotation + 1) % 4
        shape = self.rotated(rotation)

        y, x = self.tetromino_position

        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x+1)) or
                    self.fits_in_matrix(shape, (y, x-1)) or
                    self.fits_in_matrix(shape, (y, x+2)) or
                    self.fits_in_matrix(shape, (y, x-2)))
        # ^ That's how wall-kick is implemented

        if position and self.blend(shape, position):
            self.tetromino_rotation = rotation
            self.tetromino_position = position
            
            self.needs_redraw = True
            return self.tetromino_rotation
        else:
            return False
            
    def request_movement(self, direction):
        posY, posX = self.tetromino_position
        if direction == 'left' and self.blend(position=(posY, posX-1)):
            self.tetromino_position = (posY, posX-1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'right' and self.blend(position=(posY, posX+1)):
            self.tetromino_position = (posY, posX+1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'up' and self.blend(position=(posY-1, posX)):
            self.needs_redraw = True
            self.tetromino_position = (posY-1, posX)
            return self.tetromino_position
        elif direction == 'down' and self.blend(position=(posY+1, posX)):
            self.needs_redraw = True
            self.tetromino_position = (posY+1, posX)
            return self.tetromino_position
        else:
            return False

    def rotated(self, rotation=None):
        if rotation is None:
            rotation = self.tetromino_rotation
        return rotate(self.current_tetromino.shape, rotation)

    def block(self, color, shadow=False):
        colors = {'blue':   (105, 105, 255),
                  'yellow': (225, 242, 41),
                  'pink':   (242, 41, 195),
                  'green':  (22, 181, 64),
                  'red':    (204, 22, 22),
                  'orange': (245, 144, 12),
                  'cyan':   (10, 255, 226)}


        if shadow:
            end = [90] # end is the alpha value
        else:
            end = [] # Adding this to the end will not change the array, thus no alpha value

        border = Surface((BLOCKSIZE, BLOCKSIZE), pygame.SRCALPHA, 32)
        border.fill(list(map(lambda c: c*0.5, colors[color])) + end)

        borderwidth = 2

        box = Surface((BLOCKSIZE-borderwidth*2, BLOCKSIZE-borderwidth*2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                boxarr[x][y] = tuple(list(map(lambda c: min(255, int(c*random.uniform(0.8, 1.2))), colors[color])) + end) 

        del boxarr # deleting boxarr or else the box surface will be 'locked' or something like that and won't blit.
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))


        return border

    def lock_tetromino(self):
        """
        This method is called whenever the falling tetromino "dies". `self.matrix` is updated,
        the lines are counted and cleared, and a new tetromino is chosen.
        """
        self.matrix = self.blend()

        lines_cleared = self.remove_lines()
        self.lines += lines_cleared

        if lines_cleared:
            self.score += 100 * (lines_cleared**2) * self.combo

        if self.lines >= self.level*10:
            self.level += 1

        self.combo = self.combo + 1 if lines_cleared else 1

        self.set_tetrominoes()

        if not self.blend():
            self.gameover()
            
        self.needs_redraw = True

    def remove_lines(self):
        lines = []
        for y in range(MATRIX_HEIGHT):
            line = (y, [])
            for x in range(MATRIX_WIDTH):
                if self.matrix[(y,x)]:
                    line[1].append(x)
            if len(line[1]) == MATRIX_WIDTH:
                lines.append(y)

        for line in sorted(lines):
            for x in range(MATRIX_WIDTH):
                self.matrix[(line,x)] = None
            for y in range(0, line+1)[::-1]:
                for x in range(MATRIX_WIDTH):
                    self.matrix[(y,x)] = self.matrix.get((y-1,x), None)

        return len(lines)

    def blend(self, shape=None, position=None, matrix=None, shadow=False):
        """
        Does `shape` at `position` fit in `matrix`? If so, return a new copy of `matrix` where all
        the squares of `shape` have been placed in `matrix`. Otherwise, return False.
        
        This method is often used simply as a test, for example to see if an action by the player is valid.
        It is also used in `self.draw_surface` to paint the falling tetromino and its shadow on the screen.
        """
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position

        copy = dict(self.matrix if matrix is None else matrix)
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if (copy.get((y, x), False) is False and shape[y-posY][x-posX] # shape is outside the matrix
                    or # coordinate is occupied by something else which isn't a shadow
                    copy.get((y,x)) and shape[y-posY][x-posX] and copy[(y,x)][0] != 'shadow'):

                    return False # Blend failed; `shape` at `position` breaks the matrix

                elif shape[y-posY][x-posX]:
                    copy[(y,x)] = ('shadow', self.shadow_block) if shadow else ('block', self.tetromino_block)

        return copy

    def construct_surface_of_next_tetromino(self):
        shape = self.next_tetromino.shape
        surf = Surface((len(shape)*BLOCKSIZE, len(shape)*BLOCKSIZE), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.next_tetromino.color), (x*BLOCKSIZE, y*BLOCKSIZE))
        return surf

class Game(object):
    def main(self, draw_screen=True, comp_control=False):

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT)) if draw_screen else None


        self.matris = Matris(self.screen, draw_screen=draw_screen)

        # screen will always be drawn if comp_control is False
        if draw_screen or not comp_control:
            self.screen.blit(construct_nightmare(self.screen.get_size()), (0,0))
            matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
            matris_border.fill(BORDERCOLOR)
            self.screen.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))
            
            self.redraw()

        if not comp_control:
            clock = pygame.time.Clock()
            
            while True:
                try:
                    timepassed = clock.tick(50)
                    if self.matris.update((timepassed / 1000.) if not self.matris.paused else 0):
                        self.redraw()
                except GameOver:
                    return
                    
      

    def redraw(self):
        if not self.matris.paused:
            self.blit_next_tetromino(self.matris.surface_of_next_tetromino)
            self.blit_info()

            self.matris.draw_surface()

        pygame.display.flip()


    def blit_info(self):
        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 30)
        width = (WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2)) - MATRIS_OFFSET*2

        def renderpair(text, val):
            text = font.render(text, True, textcolor)
            val = font.render(str(val), True, textcolor)

            surf = Surface((width, text.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)

            surf.blit(text, text.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
            surf.blit(val, val.get_rect(top=BORDERWIDTH+10, right=width-(BORDERWIDTH+10)))
            return surf

        scoresurf = renderpair("Score", self.matris.score)
        levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        combosurf = renderpair("Combo", "x{}".format(self.matris.combo))

        height = 20 + (levelsurf.get_rect().height + 
                       scoresurf.get_rect().height +
                       linessurf.get_rect().height + 
                       combosurf.get_rect().height )

        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))

        area.blit(levelsurf, (0,0))
        area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))
        area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))

        self.screen.blit(area, area.get_rect(bottom=HEIGHT-MATRIS_OFFSET, centerx=TRICKY_CENTERX))


    def blit_next_tetromino(self, tetromino_surf):
        area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

        areasize = area.get_size()[0]
        tetromino_surf_size = tetromino_surf.get_size()[0]
        # ^^ I'm assuming width and height are the same

        center = areasize/2 - tetromino_surf_size/2
        area.blit(tetromino_surf, (center, center))

        self.screen.blit(area, area.get_rect(top=MATRIS_OFFSET, centerx=TRICKY_CENTERX))

def construct_nightmare(size):
    surf = Surface(size)

    boxsize = 8
    bordersize = 1
    vals = '1235' # only the lower values, for darker colors and greater fear
    arr = pygame.PixelArray(surf)
    for x in range(0, len(arr), boxsize):
        for y in range(0, len(arr[x]), boxsize):

            color = int(''.join([random.choice(vals) + random.choice(vals) for _ in range(3)]), 16)

            for LX in range(x, x+(boxsize - bordersize)):
                for LY in range(y, y+(boxsize - bordersize)):
                    if LX < len(arr) and LY < len(arr[x]):
                        arr[LX][LY] = color
    del arr
    return surf


if __name__ == '__main__':
    pygame.display.set_caption("MaTris")
    Game().main()
