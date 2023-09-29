import streamlit as st
from utils.network import *
from utils.webpage import *
from matplotlib import pyplot as plt

web = webpage()
node_num = web.node_num_input()
web.edge_input()
outdoor_node = web.output_node()
edge_list = web.edge_list


fig,ax = plt.subplots()
G = network(fig,ax,node_num=node_num)
G.create_edge(edge_list)
G.show(outdoor_node)


# 网页界面
st.title('Networkx Demo')
st.pyplot(fig)