from rl.agents.deterministic import DeterministicPlannerAgent
from rl_agents.agents.tree_search.deterministic import DeterministicPlannerAgent as odp
from envs.river_swim_continuous import RiverSwimContinuous
from envs.race_strategy import Race
from igraph import Graph, EdgeSeq, Edge
import plotly.graph_objects as go
import plotly.io as pio
import json
from particle_filtering.pf_mcts import PFMCTS
from pure_mcts.mcts_dpw import MCTSStochastic

def visualize(root):
    g = Graph()
    v_label = []
    a_label = []
    nr_vertices = inorderTraversal(root, g, 0, 0, v_label, a_label)
    lay = g.layout_reingold_tilford(mode="in", root=[0])
    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)
    E = [e.tuple for e in g.es]  # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    label_xs = []
    label_ys = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]
        label_xs.append((position[edge[0]][0] + position[edge[1]][0]) / 2)
        label_ys.append((2 * M - position[edge[0]][1] + 2 * M - position[edge[1]][1]) / 2)

    labels = v_label
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines',
                             line=dict(color='rgb(210,210,210)', width=1),
                             hoverinfo='none'
                             ))
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
                             name='bla',
                             marker=dict(symbol='circle-dot',
                                         size=5,
                                         color='#6175c1',  # '#DB4551',
                                         line=dict(color='rgb(50,50,50)', width=1)
                                         ),
                             text=labels,
                             hoverinfo='text',
                             opacity=0.8
                             ))

    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )
    fig.update_layout(title='Tree with Reingold-Tilford Layout',
                      annotations=make_annotations(position, v_label, label_xs, label_ys, a_label,  M, position),
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=85, t=100),
                      hovermode='closest',
                      plot_bgcolor='rgb(248,248,248)'
                      )
    fig.show()


def inorderTraversal(root, g, vertex_index, parent_index, v_label, a_label):
    if root:
        g.add_vertex(vertex_index)
        #v_label.append(str(root.index) + " Value="+str(root.V))
        v_label.append("")
        if root.parent:
            g.add_edge(parent_index, vertex_index)
            a_label.append(str(root.action) + " (" + str(root.count) + ")")
        par_index = vertex_index
        vertex_index += 1
        for i, a in enumerate(root.children.values()):
            vertex_index = inorderTraversal(a, g, vertex_index, par_index, v_label, a_label)
    return vertex_index


def make_annotations(pos, labels, Xe, Ye, a_labels, M, position, font_size=10, font_color='rgb(250,250,250)'):
    L=len(pos)
    if len(labels) != L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=labels[k], # or replace labels with a different list for the text within the circle
                x=pos[k][0]+2, y=2*M-position[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    for e in range(len(a_labels)):
        annotations.append(
            dict(
                text=a_labels[e],  # or replace labels with a different list for the text within the circle
                x=Xe[e], y=Ye[e],
                xref='x1', yref='y1',
                font=dict(color='rgb(0, 0, 0)', size=font_size),
                showarrow=False)
        )
    return annotations


if __name__ == "__main__":
    env = RiverSwimContinuous()
    alpha = 0.44
    gamma = 0.99
    budget = 100
    n_particles = 1
    max_depth = 10
    c = 1.2
    n_iters = int(budget / n_particles / max_depth)
    config = {
        "gamma": gamma,
        "budget": budget,
    }
    planner = DeterministicPlannerAgent(env, config)
    s = env.reset()
    actions = planner.plan(s)
    visualize(planner.planner.root)
    print(actions)

    s = env.reset()
    mcts_params = dict(gamma=gamma)
    mcts_params['particles'] = n_particles
    mcts = PFMCTS(root_index=s, root=None, model=None, na=env.action_space.n, ** mcts_params)
    mcts.search(n_mcts=n_iters, max_depth=max_depth, c=1.2, Env=env, mcts_env=None)
    mcts.visualize()

    s = env.reset()
    mcts_params = dict(gamma=gamma)
    mcts_params['alpha'] = alpha
    mcts = MCTSStochastic(root_index=s, root=None, model=None, na=env.action_space.n, **mcts_params)
    n_iters = int(budget / max_depth)
    mcts.search(n_mcts=n_iters, max_depth=max_depth, c=1.2, Env=env, mcts_env=None)
    mcts.visualize()



