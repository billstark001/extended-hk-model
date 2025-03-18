import msgpack
import networkx as nx
import matplotlib.pyplot as plt

def load_graph_from_msgpack(filename):
    # 读取MessagePack文件
    with open(filename, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    
    # 创建DiGraph (有向图)
    G = nx.DiGraph()
    
    # 添加图的属性
    if 'graph' in data:
        G.graph.update(data['graph'])
    
    # 添加节点和节点属性
    if 'nodes' in data:
        for node_id, attrs in data['nodes'].items():
            G.add_node(node_id, **attrs)
    
    # 添加边和边属性
    if 'adjacency' in data:
        for from_id, targets in data['adjacency'].items():
            for to_id, edge_attrs in targets.items():
                # 确保节点存在
                if from_id not in G:
                    G.add_node(from_id)
                if to_id not in G:
                    G.add_node(to_id)
                
                # 添加边及其属性
                G.add_edge(from_id, to_id, **edge_attrs)
    
    return G

# 加载图
G = load_graph_from_msgpack('graph.msgpack')

# 打印图的信息
print(f"节点数量: {G.number_of_nodes()}")
print(f"边数量: {G.number_of_edges()}")
print(f"有向图: {G.is_directed()}")

# 显示图
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=500, arrowsize=20, font_size=12)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Graph from Go")
plt.show()