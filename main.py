from model import *
from a_star import BestFirst


puzzles = ['cat', 'chicken', 'clover', 'elephant', 'fox', 'rabbit', 'reindeer', 'sailboat', 'snail2', 'telephone']


model = ModelV2('puzzles/nono-'+puzzles[0]+'.txt')


#model.draw_state(model.generate_random_state())
model.validate(model.generate_random_state())
