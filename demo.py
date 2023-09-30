import streamlit as st
from utils.network import *
from utils.webpage import *
from utils.Forward_Calculation import *
from matplotlib import pyplot as plt

node_num = node_num()
out_node = out_node(st.session_state,node_num)
edge_input(st.session_state,node_num)
edge_list = st.session_state.edge_list
pressure_dict = pressure(st.session_state,out_node)

fig,ax = plt.subplots()
P,Q = [],[]
if st.button('Calculate'):
    P,Q = VNetwork(node_num,edge_list,out_node,pressure_dict).newton_method()
    # st.write(str(P))
    # st.write(str(Q))
plot(st.session_state,node_num,edge_list,out_node,P,Q)


# 网页界面
st.title('Networkx Demo')
# st.write(str(st.session_state.G.G.edges))
# st.write(str(pressure_dict))
st.pyplot(fig)