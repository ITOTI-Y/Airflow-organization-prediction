import streamlit as st
from utils.network import *
from utils.webpage import *
from utils.Forward_Calculation import *
from matplotlib import pyplot as plt

col1,col2 = st.columns([1,1])
node_num = node_num(col1)
out_node = out_node(node_num)
edge_input(st.session_state,col2,node_num)
edge_list = st.session_state.edge_list
pressure_dict = pressure(st.session_state,out_node)

fig,ax = plt.subplots()
P,Q = [],[]
if st.button('Calculate'):
    P,Q = VNetwork(node_num,edge_list,out_node,pressure_dict).newton_method()
modify_Q_dict = plot(st.session_state,node_num,edge_list,out_node,P,Q)


# 网页界面
st.title('Networkx Demo')
col3,col4 = st.columns([1,1])
st.pyplot(fig)
st.table(table_flow(modify_Q_dict))