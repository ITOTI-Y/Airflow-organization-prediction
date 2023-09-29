import networkx as nx
import numpy as np


class network:
    node_num = 1
    edge_num = None
    options = {
    "font_size": 12,
    "node_size": 300,
    "edgecolors": "black",
    "linewidths": 2,
    "width": 2,
    "with_labels": True,
    "connectionstyle": "arc3,rad=0.1",
    # "with_labels": True,
    }

    def __init__(self,fig,ax,node_num=1) -> None:
        self.G = nx.DiGraph()
        self.fig = fig
        self.ax = ax
        self.create_nodes(node_num)
        pass

    def create_nodes(self,node_num):
        self.node_num = node_num
        self.G.add_nodes_from([i for i in range(node_num)])
    
    def color_map(self,outdoor_node=[]):
        result = []
        for i in range(self.node_num):
            if i in outdoor_node:
                result.append("red")
            else:
                result.append("white")
        self.options['node_color'] = result

    def make_edge_list(self,node_list):
        edge_list = []
        for i in range(len(node_list)):
            for j in range(i+1,len(node_list)):
                edge_list.append((node_list[i],node_list[j]))
        return edge_list
    
    def create_edge(self,edge_list):
        # self.edge_list = self.make_edge_list(node_list)
        self.G.add_edges_from(edge_list)
    
    def show(self,outdoor_node:list[int]=[]):
        pos = nx.shell_layout(self.G)
        self.color_map(outdoor_node=outdoor_node)
        nx.draw(self.G,pos,**self.options)