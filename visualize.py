import matplotlib.pyplot as plt

def visualize(instance):

    capacity_per_row = instance['capacity_per_row']
    rows_per_section = instance['rows_per_section']
    storage_products = instance['storage_products']
    storage_times = instance['storage_times']
    current_time = instance['current_time']
    type = instance['type']

    section_labels = ['A', 'B', 'C']
    section_colors = ['green', 'yellow', 'red']
    product_labels = ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5', 'Product 6']
    product_shapes = ['o', 's', '^', 'D', 'v', '*']
    product_colors = ['black', 'black', 'black', 'black', 'black', 'black']

    total_rows = sum(rows_per_section)
    total_capacity = total_rows * capacity_per_row

    # Compute section ranges (for coloring and labels)
    section_ranges = {}
    current_idx = 0
    for label, rows in zip(section_labels, rows_per_section):
        start = current_idx
        end = current_idx + rows * capacity_per_row - 1
        section_ranges[label] = (start, end)
        current_idx = end + 1

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx in range(total_capacity):
        row = idx // capacity_per_row
        col = idx % capacity_per_row

        # Determine section color
        section_color = 'gray'
        for label, (start, end) in section_ranges.items():
            if start <= idx <= end:
                section_color = section_colors[section_labels.index(label)]
                break

        rect = plt.Rectangle((col, row), 1, 1, edgecolor='black', facecolor=section_color)
        ax.add_patch(rect)

        product_id = storage_products[idx]
        storage_time = storage_times[idx]

        if product_id is not None:
            shape = product_shapes[product_id - 1]
            color = product_colors[product_id - 1]
            ax.scatter(col + 0.5, row + 0.5, s=200, c=color, marker=shape)
            if storage_time is not None:
                ax.text(col + 0.8, row + 0.8, f"{storage_time}", ha='center', va='center', fontsize=8)

    # Section labels
    for label, (start, end) in section_ranges.items():
        row_start = (start // capacity_per_row)
        row_end = (end // capacity_per_row)
        center_row = (row_start + row_end) / 2
        ax.text(-0.5, center_row + 0.5, f"Section {label}", va='center', ha='right', fontsize=10, fontweight='bold')

    # Add I/O label
    ax.text(capacity_per_row / 2, -1, 'I/O', ha='center', va='center', fontsize=12, fontweight='bold')

    # Add current time label
    ax.text(capacity_per_row + 0.5, total_rows / 2, f"Current Time: {current_time}", ha='left', va='center', fontsize=10)

    # Add type label
    ax.text(capacity_per_row + 0.5, total_rows / 2 - 1, f"Type: {type}", ha='left', va='center', fontsize=10)

    # Legend for select products
    product_ids_in_legend = [1, 2, 3, 4, 5, 6]
    legend_items = [
        plt.Line2D([0], [0], marker=product_shapes[pid - 1], color='w', markerfacecolor='black',
                   markersize=10, label=product_labels[pid - 1])
        for pid in product_ids_in_legend
    ]
    ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlim(-1, capacity_per_row + 1)
    ax.set_ylim(-2, total_rows + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax

    #plt.show()

'''
# Updated input using two vectors
instance = {
    'capacity_per_row': 5,
    'rows_per_section': [1, 1, 2],
    'storage_products': [
        1, 2, 1, None, None,
        2, 3, 1, None, None,
        4, 3, 5, 6, None,
        None, None, None, None, None,
    ],
    'storage_times': [
        1, 1, 1, None, None,
        2, 2, 2, None, None,
        3, 3, 3, 3, None,
        None, None, None, None, None
    ]
}

visualize(instance)
'''


import matplotlib.pyplot as plt

def visualize_for_gif(instance, ax):
    capacity_per_row = instance['capacity_per_row']
    rows_per_section = instance['rows_per_section']
    storage_products = instance['storage_products']
    storage_times = instance['storage_times']
    current_time = instance['current_time']
    type = instance['type']

    section_labels = ['A', 'B', 'C']
    section_colors = ['green', 'yellow', 'red']
    product_labels = ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5', 'Product 6']
    product_shapes = ['o', 's', '^', 'D', 'v', '*']
    product_colors = ['black'] * 6

    total_rows = sum(rows_per_section)
    total_capacity = total_rows * capacity_per_row

    # Compute section ranges
    section_ranges = {}
    current_idx = 0
    for label, rows in zip(section_labels, rows_per_section):
        start = current_idx
        end = current_idx + rows * capacity_per_row - 1
        section_ranges[label] = (start, end)
        current_idx = end + 1

    ax.set_xlim(-1, capacity_per_row + 1)
    ax.set_ylim(-2, total_rows + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    for idx in range(total_capacity):
        row = idx // capacity_per_row
        col = idx % capacity_per_row

        section_color = 'gray'
        for label, (start, end) in section_ranges.items():
            if start <= idx <= end:
                section_color = section_colors[section_labels.index(label)]
                break

        rect = plt.Rectangle((col, row), 1, 1, edgecolor='black', facecolor=section_color)
        ax.add_patch(rect)

        product_id = storage_products[idx]
        storage_time = storage_times[idx]

        if product_id is not None:
            shape = product_shapes[product_id - 1]
            color = product_colors[product_id - 1]
            ax.scatter(col + 0.5, row + 0.5, s=200, c=color, marker=shape)
            if storage_time is not None:
                ax.text(col + 0.8, row + 0.8, f"{storage_time}", ha='center', va='center', fontsize=8)

    for label, (start, end) in section_ranges.items():
        row_start = start // capacity_per_row
        row_end = end // capacity_per_row
        center_row = (row_start + row_end) / 2
        ax.text(-0.5, center_row + 0.5, f"Section {label}", va='center', ha='right', fontsize=10, fontweight='bold')

    ax.text(capacity_per_row / 2, -1, 'I/O', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(capacity_per_row + 0.5, total_rows / 2, f"Current Time: {current_time}", ha='left', va='center', fontsize=10)
    ax.text(capacity_per_row + 0.5, total_rows / 2 - 1, f"Type: {type}", ha='left', va='center', fontsize=10)

    # Legend
    legend_items = [
        plt.Line2D([0], [0], marker=product_shapes[i], color='w', markerfacecolor='black',
                   markersize=10, label=product_labels[i]) for i in range(6)
    ]
    ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1, 1))
