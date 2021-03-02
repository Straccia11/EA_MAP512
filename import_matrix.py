with open('matrix/adj_matrix.txt', 'r') as f:
    l = [[int(num) for num in line.split(' ')] for line in f]

m = np.array(l)

g = Graph(directed=False)

g.add_edge_list(np.transpose(m.nonzero()))

graph_draw(g, pos=sfdp_layout(g, cooling_step=0.99),vertex_text=g.vertex_index)
