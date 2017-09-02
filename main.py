from model import *
from a_star import BestFirst


puzzles = ['cat', 'chicken', 'clover', 'elephant', 'fox', 'rabbit', 'reindeer', 'sailboat', 'snail2', 'telephone']


model = Model('puzzles/nono-'+puzzles[0]+'.txt')

for v in model.row_variables[0]:
    print(v.domain)
