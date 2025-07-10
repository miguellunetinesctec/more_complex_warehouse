import matplotlib.pyplot as plt
import random
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib as mpl


def generate_warehouse(number_of_blocks, number_of_aisles, number_of_locations_per_aisle, in_out, double_aisle=False, plot=True):

    #Set dimensions of the warehouse
    square_size = 4 # size of each square
    spacing_x = 8  # horizontal spacing between aisles
    spacing_y = 4  # vertical spacing between rows
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(120, 8))

    #Create graph with networkx
    G = nx.Graph()
    
    warehouse = {}
    location_id = 0

    in_out_node = (0, in_out)

    # Create In/Out retangle in the bottom center
    if double_aisle:
        x = square_size + spacing_x/2 + in_out*(spacing_x/2+square_size)
    else:
        x = square_size + spacing_x/2 + in_out*(spacing_x+square_size)/2
    y = -spacing_y/2
    rect = plt.Rectangle((x - square_size, y - square_size/2), 2*square_size, square_size, fill=False, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x, y, 'In/Out', fontsize=8, ha='center', va='center')

    if double_aisle:
        warehouse[location_id] = {
            'location_id': location_id,
            'location_coords': (x, y),
            'node_id': in_out_node}
    else:
        warehouse[location_id] = {
            'location_id': location_id,
            'location_coords': (x, y),
            'node_id': [in_out_node]}

    location_id += 1
    col_id = 0


    for aisle in range(number_of_aisles if double_aisle else number_of_aisles+1):
        
        #Add first row node
        row_id = 0
        if double_aisle:
            x_node = aisle*(spacing_x+2*square_size) + spacing_x/2 + square_size
        else:
            x_node = aisle*(spacing_x+square_size) + spacing_x/2 + square_size
        y_node = spacing_y/2
        if aisle < number_of_aisles:
            G.add_node((row_id,col_id), pos=(x_node, y_node))

        if double_aisle:

            for col in range(2):
                
                # Start adding nodes from the second row
                row_id = 1

                # Define if the storage location is on the left or right side of the aisle
                if col == 0:
                    tune_x = -spacing_x/2-square_size/2
                else:
                    tune_x = +spacing_x/2+square_size/2

                for block in range(number_of_blocks):
                    
                    for row in range(1,number_of_locations_per_aisle+1):
                    
                        y_node = block * (number_of_locations_per_aisle*square_size + spacing_y) + row * square_size - square_size/2 + spacing_y

                        if col == 0:
                            G.add_node((row_id, col_id), pos=(x_node, y_node))
                        
                        x = aisle * (spacing_x+2*square_size) + tune_x + spacing_x/2 + square_size
                        y = block * (number_of_locations_per_aisle*square_size + spacing_y) + row * square_size - square_size/2 + spacing_y 

                        warehouse[location_id] = {
                            'location_id': location_id,
                            'location_coords': (x, y),
                            'node_id': (row_id, col_id),
                            'node_coords': (x_node, y_node)
                        }

                        row_id += 1

                        # Draw square
                        rect = plt.Rectangle((x - square_size/2, y - square_size/2), square_size, square_size, fill=False, edgecolor='black')
                        ax.add_patch(rect)
                        # Label ID
                        ax.text(x, y, str(location_id), fontsize=8, ha='center', va='center')

                        location_id += 1

                    y_node = block * (number_of_locations_per_aisle*square_size + spacing_y) + number_of_locations_per_aisle * square_size + spacing_y*3/2
                    G.add_node((row_id,col_id), pos=(x_node, y_node))
                    row_id += 1
        else:

            tune_x = -spacing_x/2-square_size/2
            row_id = 1

            for block in range(number_of_blocks):
                    
                    for row in range(1,number_of_locations_per_aisle+1):
                    
                        y_node = block * (number_of_locations_per_aisle*square_size + spacing_y) + row * square_size - square_size/2 + spacing_y

                        if aisle < number_of_aisles:
                            G.add_node((row_id, col_id), pos=(x_node, y_node))
                        
                        x = aisle * (spacing_x+square_size) + tune_x + spacing_x/2 + square_size
                        y = block * (number_of_locations_per_aisle*square_size + spacing_y) + row * square_size - square_size/2 + spacing_y 

                        if aisle == 0:
                            node_ids = [(row_id, col_id)]
                        elif aisle == number_of_aisles:
                            node_ids = [(row_id, col_id)]
                        else:
                            node_ids = [(row_id, col_id), (row_id, col_id-2)]

                        warehouse[location_id] = {
                            'location_id': location_id,
                            'location_coords': (x, y),
                            'node_id': node_ids,
                            'node_coords': [G.nodes[node]['pos'] for node in node_ids]
                        }

                        row_id += 1

                        # Draw square
                        rect = plt.Rectangle((x - square_size/2, y - square_size/2), square_size, square_size, fill=False, edgecolor='black')
                        ax.add_patch(rect)
                        # Label ID
                        ax.text(x, y, str(location_id), fontsize=8, ha='center', va='center')

                        location_id += 1

                    if aisle < number_of_aisles:
                        y_node = block * (number_of_locations_per_aisle*square_size + spacing_y) + number_of_locations_per_aisle * square_size + spacing_y*3/2
                        G.add_node((row_id,col_id), pos=(x_node, y_node))
                        row_id += 1

    


        if aisle < (number_of_aisles-1):
            
            col_id += 1
            row_id = 0
            if double_aisle:
                x_node = aisle * (spacing_x+2*square_size) + spacing_x + 2*square_size
            else:
                x_node = aisle * (spacing_x+square_size) + spacing_x + square_size*3/2
            y_node = spacing_y/2
            G.add_node((row_id,col_id), pos=(x_node, y_node))
            for block in range(number_of_blocks):
                row_id = (block+1) * (number_of_locations_per_aisle+1)
                y_node = block * (number_of_locations_per_aisle*square_size + spacing_y) + number_of_locations_per_aisle * square_size + spacing_y*3/2
                G.add_node((row_id,col_id), pos=(x_node, y_node))
            
            col_id += 1
    

    #Write Block and the number in the left side
    for block in range(number_of_blocks):
        x = -spacing_x
        y = number_of_locations_per_aisle*square_size/2 + block * (number_of_locations_per_aisle*square_size + spacing_y) + spacing_y
        ax.text(x, y, f"Block {block+1}", fontsize=8, ha='center', va='center')
    
    for aisle in range(number_of_aisles):
        if double_aisle:
            x = aisle*(spacing_x + 2*square_size) + spacing_x/2 + square_size
        else:
            x = aisle*(spacing_x + square_size) + spacing_x/2 + square_size
        y = number_of_blocks*(number_of_locations_per_aisle*square_size + spacing_y)  + spacing_y
        ax.text(x, y, f"Aisle {aisle+1}", fontsize=8, ha='center', va='center')

    #Create the edges between the nodes
    #Connect the nodes in the same aisle (vertical connections)
    
    for col_id in range(0, number_of_aisles * 2, 2):  # Adjusted range to include all columns
        for row_id in range(number_of_blocks * (number_of_locations_per_aisle + 1)):  # Adjusted range to include all rows
            if (row_id, col_id) in G.nodes and (row_id + 1, col_id) in G.nodes:
                G.add_edge((row_id, col_id), (row_id + 1, col_id), weight=1)  # Set the distance (weight) of the edge to be 1
    
    # Connect the nodes in the same cross aisle (horizontal connections)
    for row_id in range(0, number_of_blocks * (number_of_locations_per_aisle + 1) + 1, number_of_locations_per_aisle + 1):
        for col_id in range(number_of_aisles*2):
            if (row_id, col_id) in G.nodes and (row_id, col_id + 1) in G.nodes:
                G.add_edge((row_id, col_id), (row_id, col_id + 1),weight = 1)  # Edges in NetworkX are bidirectional by default for undirected graphs

    #Plot the edges that were created
    for edge in G.edges():
        x1, y1 = G.nodes[edge[0]]['pos']
        x2, y2 = G.nodes[edge[1]]['pos']
        ax.plot([x1, x2], [y1, y2], 'k-', lw=0.5)

    #Plot a red dot in the nodes
    for node in G.nodes():
        pos = G.nodes[node]['pos']
        ax.plot(pos[0], pos[1], 'ro', markersize=1)
        ax.text(pos[0], pos[1], str(node), fontsize=8, ha='center', va='center')

    if double_aisle:
        ax.set_xlim(-1,number_of_aisles * (spacing_x+2*square_size) + 1)
    else:
        ax.set_xlim(-1,number_of_aisles * (spacing_x+square_size)+square_size + 1)
    ax.set_ylim(-spacing_y/2-square_size/2-1, number_of_blocks * (number_of_locations_per_aisle * square_size + spacing_y) + spacing_y/2 + 1)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.axis('off') 
    #if plot:
     #   plt.show()

    return warehouse, G

# Given 2 node_ids (row_id, col_id), return the distance between them
def get_distance(location_id_1, location_id_2, warehouse, graph, double_aisle = True, distance_type='path'):

    if double_aisle:

        node_id_1 = warehouse[location_id_1]['node_id']
        node_id_2 = warehouse[location_id_2]['node_id']

        if distance_type == 'manhattan':
            x1, y1 = warehouse[location_id_1]['node_id']
            x2, y2 = warehouse[location_id_2]['node_id']
            distance = abs(x2 - x1) + abs(y2 - y1)
        
        elif distance_type == 'path':
            distance = nx.shortest_path_length(graph, source=node_id_1, target=node_id_2)
            #Print the follwed route
            #path = nx.shortest_path(graph, source=node_id_1, target=node_id_2)
            #print("Shortest path:", path)
        
    else:

        possible_node_ids_1 = warehouse[location_id_1]['node_id']
        possible_node_ids_2 = warehouse[location_id_2]['node_id']
        all_distances = []

        for node_id_1 in possible_node_ids_1:
            for node_id_2 in possible_node_ids_2:
                if distance_type == 'manhattan':
                    distance = abs(node_id_1[0] - node_id_2[0]) + abs(node_id_1[1] - node_id_2[1])
                    all_distances.append(distance)
                elif distance_type == 'path':
                    distance = nx.shortest_path_length(graph, source=node_id_1, target=node_id_2)
                    all_distances.append(distance)
        
        distance = min(all_distances)

    return distance

def assign_sections(warehouse, in_out_id, graph, double_aisle=False):

    number_of_locations = len(warehouse) - 1  # Exclude the in/out node

    sections = {}
    distances = []
    assignments = []

    # Calculate all distances from locations to the in/out node
    for location_id in range(1, number_of_locations + 1):
        distance = get_distance(location_id, in_out_id, warehouse, graph, double_aisle=double_aisle, distance_type='path')
        distances.append(distance)

    # Determine thresholds based on min and max distances
    min_distance = min(distances)
    max_distance = max(distances)
    range_distance = max_distance - min_distance

    threshold_a = min_distance + 0.2 * range_distance
    threshold_b = min_distance + 0.4 * range_distance

    # Assign sections based on thresholds
    for location_id, distance in zip(range(1, number_of_locations + 1), distances):
        if distance <= threshold_a:
            sections[location_id] = 'A'
        elif distance <= threshold_b:
            sections[location_id] = 'B'
        else:
            sections[location_id] = 'C'
        assignments.append(sections[location_id])

    return sections

def visualize_warehouse(current_time, warehouse, G, product_types, location_types, sections, id_vector, product_vector, dos_vector, location_type_vector, ax=None):

    min_dos_vector = 1
    max_dos_vector = 10
    
    square_size = 4
    # Set up plot
    # Create fig/ax if not passed (so old code still works)
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_title(f"Warehouse Layout at Time {current_time}", fontsize=12)

    # Define section colors
    section_colors = {'A': '#D3F8D3', 'B': '#FFFACD', 'C': '#FFD1D1'}

    # Define product shapes
    product_shapes = ['o', '^', 's'] #'D', 'p', '*', 'h', 'H', 'v', '<', '>']

    square_size = 4

    order_locations_by_section = {section: [] for section in section_colors.keys()}
    for location_id, section in sections.items():
        order_locations_by_section[section].append(location_id)

    # Draw each location square and product shape
    for location_id in [0] + order_locations_by_section['C'] + order_locations_by_section['B'] + order_locations_by_section['A']:

        if location_id == 0:
            x, y = warehouse[location_id]['location_coords']
            # Draw rectangle with "In and Out" label
            rect = plt.Rectangle((x - square_size/2, y - square_size / 4), square_size, square_size/2,
                                facecolor='white', edgecolor='black', linewidth=1)
            ax.add_patch(rect)

            # Label the rectangle with "In and Out"
            ax.text(x, y, "In & Out", fontsize=8, ha='center', va='center')
            
        else:

            task_id = id_vector[location_id-1]
            product_id = product_vector[location_id-1]
            dos = dos_vector[location_id-1]
            x, y = warehouse[location_id]['location_coords']

            # Get section border color and background color
            section = sections.get(location_id, 0)  # Default to 'C' if not found
            #background_color = section_colors.get(section, 'black')
            import numpy as np
            # Create a gradient of colors based on the total number of locations
            total_locations = max(location_types.keys()) + 1
            gradient_colors = plt.cm.viridis(np.linspace(0, 1, total_locations))

            # Assign background color based on location type
            location_type = location_type_vector[location_id - 1]
            background_color = gradient_colors[location_type]


            # Draw square with background color and colored border
            rect = plt.Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size,
                                facecolor=background_color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)

            if task_id != None:
                # Assign color based on duration of stay (dos)
                color = plt.cm.RdYlGn(1 - dos / max_dos_vector)  # Red for higher, green for lower
                #color = 'green'

                # Draw product shape inside the square
                shape = product_shapes[product_id % len(product_shapes)]
                #shape = 'o'  # Circle for all products
                ax.scatter(x, y-square_size/8, c=[color], marker=shape, s=1000, edgecolors='black', linewidths=0.5)
                ax.text(x, y-square_size/8, str(task_id), fontsize=8, ha='center', va='center')

            #Plot a little white retangle in the top of the square
            rect = plt.Rectangle((x - square_size / 2, y + 0.50 * square_size / 2), square_size, square_size / 4,
                                facecolor='white', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            # Label ID
            ax.text(x - 0.75 * square_size / 2, y + 0.75 * square_size / 2, str(location_id), fontsize=8, ha='center', va='center')
            ax.text(x + 0.75 * square_size / 2, y + 0.75 * square_size / 2, str(location_types[location_type_vector[location_id-1]]['name']), fontsize=8, ha='center', va='center')
            
    # Plot edges
    for edge in G.edges():
        x1, y1 = G.nodes[edge[0]]['pos']
        x2, y2 = G.nodes[edge[1]]['pos']
        ax.plot([x1, x2], [y1, y2], 'k-', lw=0.5)

    # Plot nodes as red dots
    for node in G.nodes():
        pos = G.nodes[node]['pos']
        ax.plot(pos[0], pos[1], 'ro', markersize=1)
        ax.text(pos[0], pos[1], str(node), fontsize=8, ha='center', va='center')

    # Set axis limits
    all_x = [loc['location_coords'][0] for loc in warehouse.values()] + [pos[0] for pos in nx.get_node_attributes(G, 'pos').values()]
    all_y = [loc['location_coords'][1] for loc in warehouse.values()] + [pos[1] for pos in nx.get_node_attributes(G, 'pos').values()]
    ax.set_xlim(-1, max(all_x) + square_size / 2 + 1)
    ax.set_ylim(min(all_y) - square_size/2 - 1, max(all_y) + 1)

    ax.set_aspect('equal')
    ax.grid(False)
    ax.axis('off')  # Hide the axes

    # After plotting everything else, add this at the end before plt.show()

    '''
    # 1) Section legend (colors)
    section_patches = [mpatches.Patch(color=color, label=f'Section {label}') 
                    for label, color in section_colors.items()]

    # 2) Product legend (shapes)
    product_patches = [mlines.Line2D([], [], color='black', marker=shape, linestyle='None',
                                    markersize=10, label=f'Product {i}') 
                    for i, shape in enumerate(product_shapes)]
    '''

    #Plot the color gradient for the location types
    # Create a color gradient for the location types

    # Create a color gradient for the location types
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=len(location_types)-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for ScalarMappable
    # Add colorbar for location types
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.07)
    cbar.set_label('Location Types', fontsize=9)
    cbar.set_ticks(range(len(location_types)))
    cbar.set_ticklabels([location_types[index]['name'] for index in location_types.keys()])
    
    # 3) DOS gradient legend (continuous)
    norm = mpl.colors.Normalize(vmin=min_dos_vector, vmax=max_dos_vector)
    cmap = plt.cm.RdYlGn_r
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for ScalarMappable

    # Add colorbar for DOS gradient
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.00)
    cbar.set_label('Expected Duration of Stay (DOS)', fontsize=9)
    
    #plt.show()

'''
# Example usage:
number_of_blocks = 1
number_of_aisles = 5
number_of_locations_per_aisle = 5
in_out = 4
double_aisle = False

# Generate warehouse and graph
warehouse, G = generate_warehouse(number_of_blocks=number_of_blocks, number_of_aisles=number_of_aisles, number_of_locations_per_aisle=number_of_locations_per_aisle, in_out=in_out, double_aisle=double_aisle)
sections = assign_sections(warehouse, 0, G, double_aisle=False)

id_vector = [random.randint(1, 100) for _ in range(len(warehouse)-1)]
product_vector = [random.randint(0, 2) for _ in range(len(warehouse)-1)]
dos_vector = [random.randint(1, 10) for _ in range(len(warehouse)-1)]
visualize_warehouse(0, warehouse, G, sections, product_vector=product_vector, id_vector=id_vector, dos_vector=dos_vector)
'''

import os
from PIL import Image

def save_warehouse_frames(warehouse, warehouse_states, G, product_types, location_types, sections, output_folder="frames"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_paths = []
    
    for t, state in enumerate(warehouse_states):
        # Unpack state
        id_vector, product_vector, dos_vector, location_type_vector = state
        
        # Call your existing visualize function but save the figure instead of showing
        fig, ax = plt.subplots(figsize=(16, 9))
        visualize_warehouse(
            current_time=t,
            warehouse=warehouse,
            G=G,
            product_types=product_types,
            location_types=location_types,
            sections=sections,
            id_vector=id_vector,
            product_vector=product_vector,
            dos_vector=dos_vector,
            location_type_vector=location_type_vector,
            ax=ax  # pass ax so we don't call plt.show()
        )
        frame_path = os.path.join(output_folder, f"frame_{t:02d}.png")
        plt.savefig(frame_path, bbox_inches='tight')
        plt.close(fig)
        image_paths.append(frame_path)
    
    return image_paths

def create_gif_from_frames(image_paths, output_path="warehouse_animation.gif", duration=500):
    images = [Image.open(path) for path in image_paths]
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    return output_path

'''
# Example usage:
number_of_blocks = 1
number_of_aisles = 5
number_of_locations_per_aisle = 5
in_out = 4
double_aisle = False

# Generate warehouse and graph
warehouse, G = generate_warehouse(number_of_blocks=number_of_blocks, number_of_aisles=number_of_aisles, number_of_locations_per_aisle=number_of_locations_per_aisle, in_out=in_out, double_aisle=double_aisle)
sections = assign_sections(warehouse, 0, G, double_aisle=False)

# Suppose you have multiple time steps (say, T=10):
T = 10
warehouse_states = []
for t in range(T):
    # For each time step, create the vectors for that state (randomly for this example)
    id_vector_t = [random.randint(1, 100) for _ in range(len(warehouse)-1)]
    product_vector_t = [random.randint(0, 2) for _ in range(len(warehouse)-1)]
    dos_vector_t = [random.randint(1, 10) for _ in range(len(warehouse)-1)]
    location_type_t = [random.randint(0, 2) for _ in range(len(warehouse)-1)]
    warehouse_states.append((id_vector_t, product_vector_t, dos_vector_t, location_type_t))


product_names_vector = ["P1", "P2", "P3"]
allowed_location_types_vector = [
    ["L1", "L2", "L3"],
    ["L1", "L2"],
    ["L1", "L3"]
]
mean_size_vector = [1, 2, 3]
std_size_vector = [0, 0, 0]
mean_dos_vector = [1, 2, 3]
std_dos_vector = [0, 0, 0]

location_names_vector = ["L1", "L2", "L3", "L4"]
incompatible_location_names_vector = [
    ["L2", "L3"],
    ["L1", "L3"],
    ["L1", "L2"]
]
# Generate location types

from instance_generator import generate_product_types, generate_location_types
location_types = generate_location_types(location_names_vector, incompatible_location_names_vector)
product_types = generate_product_types(location_types, product_names_vector, allowed_location_types_vector, mean_size_vector, std_size_vector, mean_dos_vector, std_dos_vector)


# Save frames
frame_paths = save_warehouse_frames(warehouse, warehouse_states, G, product_types, location_types, sections)
gif_path = create_gif_from_frames(frame_paths)
print(f"\n🎞️  GIF created at: {gif_path}")
'''
