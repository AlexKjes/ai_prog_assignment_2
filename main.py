from model import *
from a_star import BestFirst
import csp_graph



puzzles = ['cat', 'chicken', 'clover', 'elephant', 'fox', 'rabbit', 'reindeer', 'sailboat', 'snail2', 'telephone']


model = Model('puzzles/nono-'+puzzles[0]+'.txt')

def assume(i=0):
    #print(i)
    if model.is_solution():
        print("Solution")
        return True
    if model.is_invalid():
        return False
    for n in model.csp_graph.nodes:
        if len(n.domain) > 1:
            for c in n.domain:
                model.csp_graph.save_state()
                n.update([c])
                ret = assume(i+1)
                if ret:
                    return True
                model.csp_graph.revert()

#assume()

for y in model.row_variables:
    print([v.domain for v in y])

model.draw_state(model.fill_matrix())
input()

def  test():
    a = model.row_variables
    b = a[7]
    n = b[1]

    print("Before: ")
    for v in b:
        print(v.domain)


    print("After: ")

    n.update([n.domain[-1]])
    for v in b:
        print(v.domain)
