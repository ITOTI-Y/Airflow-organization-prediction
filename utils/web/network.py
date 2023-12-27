import networkx as nx
import numpy as np
from copy import copy


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

    def __init__(self,node_num=1) -> None:
        self.G = nx.DiGraph()
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
    
    def show(self,outdoor_node:list[int]=[],P_dict=None,Q_dict=None):
        pos = nx.shell_layout(self.G)
        self.color_map(outdoor_node=outdoor_node)
        if Q_dict:
            keys = list(Q_dict.keys())
            values = list(Q_dict.values())
            new_edge_list = []
            self.G.clear_edges()
            for i in range(len(values)):
                if float(values[i]) <= 0:
                    Q_dict[keys[i]] = abs(float(values[i]))
                    new_edge_list.append(keys[i][::-1])
                else:
                    new_edge_list.append(keys[i])
            self.G.add_edges_from(new_edge_list)
            nx.draw_networkx_edge_labels(self.G,pos,edge_labels=Q_dict)
        if P_dict:
            custom_label_pos = {}
            for node, (x, y) in pos.items():
                if node == 0:
                    custom_label_pos[node] = (x, y-0.15)
                    continue
                elif y > pos[0][1] + 0.15:
                    custom_label_pos[node] = (x, y+0.15) 
                    continue
                elif y < pos[0][1] - 0.15:
                    custom_label_pos[node] = (x, y-0.15)
                    continue
                else:
                    custom_label_pos[node] = (x+0.15, y)
            

            # custom_label_pos = {node: (x, y-0.15) for node, (x, y) in pos.items()}
            nx.draw_networkx_labels(self.G,custom_label_pos,labels=P_dict,font_color="red")
        nx.draw(self.G,pos,**self.options)