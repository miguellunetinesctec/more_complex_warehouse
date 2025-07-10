import random

def generate_product_types(location_types, product_types_vector, allowed_location_types_vector, mean_size_vector, std_size_vector, mean_dos_vector, std_dos_vector):

    product_types_vector = ["D1", "D2", "D3"]
    mean_size_vector = [1, 1, 1]
    std_size_vector = [0, 0, 0]
    mean_dos_vector = [2, 4, 7]
    std_dos_vector = [0.5, 1, 2]

    num_product_types = len(product_types_vector)

    product_types = {}

    for i in range(num_product_types):
        product_types[i] = {'name': product_types_vector[i],
                    'allowed_location_types': [next(key for key, value in location_types.items() if value['name'] == loc) for loc in allowed_location_types_vector[i]],
                    'mean_size': mean_size_vector[i],
                    'std_size': std_size_vector[i],
                    'mean_dos': mean_dos_vector[i],
                    'std_dos': std_dos_vector[i],}
    
    return product_types

def generate_location_types(location_types_vector, incompatible_location_types_vector):

    num_location_types = len(location_types_vector)

    location_types = {}

    for i in range(num_location_types):
        location_types[i] = {'name': location_types_vector[i]}

    for i in range(num_location_types):
        incompatible_location_types = [next(key for key, value in location_types.items() if value['name'] == loc) for loc in incompatible_location_types_vector[i]]
        location_types[i]['incompatible_location_types'] = incompatible_location_types
                             
    return location_types

def generate_instances(product_types, num_days, avg_orders_per_day, st_orders_per_day, seed):

    instances = []

    task_id = 0
    for arrival_day in range(1, num_days + 1):
        orders_per_day = int(random.normalvariate(avg_orders_per_day, st_orders_per_day))
        for _ in range(orders_per_day):
            product_type = random.choice(range(len(product_types)))
            size = 1 #max(1, int(random.normalvariate(product_types[product_type]['mean_size'], product_types[product_type]['std_size'])))
            departure_date = arrival_day + max(1, int(random.normalvariate(product_types[product_type]['mean_dos'], product_types[product_type]['std_dos'])))
            instances.append([task_id, arrival_day, size, product_type, departure_date])
            task_id += 1

    #Store the instances in json format
    import json
    with open(f'instances/instances_seed_{seed}.json', 'w') as f:
        json.dump(instances, f, indent=4)
    # Store the instances in CSV format

    import pandas as pd
    df = pd.DataFrame(instances, columns=['Id', 'Day', 'Size', 'Product Type', 'Departure Date'])
    df.to_csv(f'instances/instances_seed_{seed}.csv', index=True)

    return instances

'''
location_names = ["L1", "L2", "L3"]
incompatible_location_names = [
    ["L2", "L3"],
    ["L1", "L3"],
    ["L1", "L2"]
]
location_types = generate_location_types(location_names, incompatible_location_names)


allowed_location_types_vector = [
    ["L1", "L2", "L3"],
    ["L1", "L2"],
    ["L1", "L3"]
]
# Example usage
num_product_types = 3
product_types = generate_product_types(location_types, ["D1", "D2", "D3"], allowed_location_types_vector, [1, 1, 1], [0, 0, 0], [2, 4, 7], [0.5, 1, 2])
# Generate instances for 5 days with 3 orders per day
num_days = 10
avg_orders_per_day = 3
st_orders_per_day = 0

for seed in range(1, 6):
    instances = generate_instances(product_types, num_days, avg_orders_per_day, st_orders_per_day, seed)
'''

'''
#Read the instances from the json file
import json
with open('instances.json', 'r') as f:
    instances = json.load(f)
    print("Instances loaded from JSON file:")
    for day, orders in instances.items():
        print(f"Day {day}: {orders}")
#Read the instances from the CSV file
import pandas as pd
df = pd.read_csv('instances.csv', index_col=0)
print("Instances loaded from CSV file:")
print(df)
'''