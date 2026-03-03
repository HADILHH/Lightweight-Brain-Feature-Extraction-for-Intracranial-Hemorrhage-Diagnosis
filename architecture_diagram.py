from graphviz import Digraph

# =============================
# Create Digraph
# =============================
dot = Digraph(comment='Neural Network Architecture', format='png')

# Input layer
dot.node('I', 'Input Features\n(lbp_mean, lbp_var,\ncontrast, energy,\nhomogeneity, correlation)')

# Hidden layer
dot.node('H', 'Hidden Layer\n(8 neurons, ReLU)')

# Output layer
dot.node('O', 'Output Layer\n(1 neuron, Sigmoid)')

# Prediction
dot.node('P', 'Prediction\nNormal / Hemorrhage')

# Connect nodes
dot.edges(['IH', 'HO', 'OP'])

# =============================
# Render Diagram
# =============================
output_path = 'architecture_diagram'
dot.render(output_path, view=True)  # Creates architecture_diagram.png and opens it
print(f"✅ Diagram saved as {output_path}.png")
