from model import *
from a_star import BestFirst
import csp_graph
import graph


puzzles = ['test', 'cat', 'chick', 'clover', 'elephant', 'fox', 'rabbit', 'reindeer', 'sailboat', 'snail2', 'telephone']


model = Model('puzzles/nono-'+puzzles[9]+'.txt')
search_graph = graph.Graph(model.generate_state_id(), model.generate_next)
solver = BestFirst(search_graph, model.h, model.is_solution)




i = 0
for n in model.csp_graph.nodes:
    i += len(n.domain)
print(i)
if i != 0:
    model.draw_state(model.generate_state_id())

#for rp in model.column_sequences[6]:
#    print(rp)

model.csp_graph.load_state_by_key(model.pre_key)
input()


while solver.solution == None:

    solver.next()
    model.draw_state(solver.last_expanded.state)
    #print("g: {}, h: {}".format(solver.last_expanded.g, solver.last_expanded.h))

model.draw_state(solver.last_expanded.state)
print("solution found")
input()
"""
def assume(i=0):
    print(i)
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


"""

"""
assume()

for y in model.column_variables:
    print([v.domain for v in y])

model.draw_state(model.fill_matrix())
input()
"""

def  test():
    a = model.column_variables
    b = a[7]
    #n = b[1]

    print("Before: ")
    for v in a:
        print([w.domain for w in v])
    model.generate_2d_row_constraints()
    model.generate_2d_col_constraints()
    print("After: ")


    for v in a:
        print([w.domain for w in v])

#test()