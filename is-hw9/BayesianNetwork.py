import networkx as nx
import numbers
import pprint
import matplotlib.pyplot as plt
from functools import reduce


pp = pprint.PrettyPrinter(indent=4)


class BayesianNetwork(object):
    def __init__(self):
        self.variables = {}
        self.G = nx.DiGraph()
        self.CPD = {}

    def add_variable(self, variable, parent=None):
        self.G.add_node(variable)
        if parent is not None and parent in self.G.nodes:
            self.G.add_edge(parent, variable)

    def add_cpd(self, variable, table):
        # Todo: verify if the table corresponds to the variable
        if variable not in self.G.nodes:
            print('%s variable doesn\'t exist.', variable)
            print('Valids are:', self.G.nodes)
            return
        self.CPD[variable] = table

    def plot(self):
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos)
        nx.draw_networkx_labels(self.G, pos)

        plt.show()

    def infer(self, variable):
        # cpd_variable = self.CPD[variable]
        # if isinstance(cpd_variable[0], numbers.Number):
        #     return cpd_variable
        parents = self.G.in_edges(variable)
        if not parents:
            return self.CPD.get(variable, [])

        cpd_parents = []
        for parent, _ in parents:
            cpd_parents.append(self.infer(parent))
        cpd = self.CPD.get(variable)

        print('cpd_parents', cpd_parents)
        print('cpd', cpd)
        print(list(zip(*cpd)))
        # for parent in cpd_parents
        # parent_factors = zip(*cpd_parents)
        # print(list(zip(*cpd_parents)))

        # for values in cpd:
        #     for x in values:
        #         for a, b in zip(*cpd_parents)
        #         print('x', x)
        return []
        # print(parents)
        # for parent in parents:

        # return self.CPD[variable]
        # print(self.CPD['variable'])
        # if len(self.CPD[variable]) == 1:
        #     return self.CPD[variable]
        # #
        # return []


# if __name__ == '__main__':
x = BayesianNetwork()
x.add_variable('diff')
x.add_variable('intel')
x.add_variable('grade', 'intel')
x.add_variable('grade', 'diff')
x.add_variable('letter', 'grade')
x.add_variable('sat', 'intel')

x.add_cpd('diff', [0.6, 0.4])
x.add_cpd('intel', [0.7, 0.3])
x.add_cpd('sat', [[0.95, 0.05], [0.2, 0.8]])
x.add_cpd(
    'grade',
    [
        # [
            [0.3, 0.4, 0.3],
            [0.05, 0.25, 0.7],
        # ],
        # [
            [0.9, 0.08, 0.02],
            [0.5, 0.3, 0.2]
        # ],
    ]
)
x.add_cpd('letter', [[0.1, 0.9], [0.4, 0.6], [0.99, 0.01]])

pp.pprint(x.CPD)

infer = x.infer('grade')
print(infer)
# print("x.infer('diff')", )
