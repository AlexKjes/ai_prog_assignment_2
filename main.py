from model import *
from a_star import BestFirst


puzzles = ['cat', 'chicken', 'clover', 'elephant', 'fox', 'rabbit', 'reindeer', 'sailboat', 'snail2', 'telephone']


model = ModelV1('puzzles/nono-'+puzzles[0]+'.txt')



print(model.generate_random_state())
