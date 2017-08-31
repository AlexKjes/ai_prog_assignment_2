from model import Model
from a_star import BestFirst


puzzles = ['cat', 'chicken', 'clover', 'elephant', 'fox', 'rabbit', 'reindeer', 'sailboat', 'snail2', 'telephone']


model = Model('puzzles/nono-'+puzzles[0]+'.txt')

for d in model.domains[1][-1]:
    print(d)
