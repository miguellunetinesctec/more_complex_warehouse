import gurobipy as gp
from gurobipy import GRB
from miscelaneous import generate_warehouse, get_distance, assign_sections, save_warehouse_frames, create_gif_from_frames
import json
import pandas as pd

def lp(warehouse, G, product_types, location_types, first_locations, instance, current_time, double_aisle=False, policy='phs', max_reallocations = 0, theta_cfa=0.5):
        
    # Sets

    number_of_tasks = len(instance)
    TaskSet = range(number_of_tasks)

    number_of_storage_locations = len(warehouse)
    LocationSet = range(number_of_storage_locations)
    StorageLocationSet = range(1, number_of_storage_locations)

    PeriodSet = range(current_time, max([task[4] for task in instance]) + 1)  # Assuming task[4] is the last period

    ProductTypeSet = [index for index, item in product_types.items()]

    LocationTypeSet = [location_id for location_id, item in location_types.items()]

    Compatibility = {(p, u): 0 for p in ProductTypeSet for u in LocationTypeSet}
    for product in product_types:
        for location in product_types[product]['allowed_location_types']:
            Compatibility[product, location] = 1
    
    IncompatibleSet = []
    for location in location_types:
        for incompatible_location in location_types[location]['incompatible_location_types']:
            if location != incompatible_location and (location, incompatible_location) not in IncompatibleSet:
                IncompatibleSet.append((location, incompatible_location))
    
    # Parameters
    Arrival = {Task: 0 for Task in TaskSet}
    Departure = {Task: 0 for Task in TaskSet}
    TaskSize = {Task: 0 for Task in TaskSet}
    ProductType = {Task: 0 for Task in TaskSet}
    In = {Location: 0 for Location in TaskSet}
    Out = {Location: 0 for Location in TaskSet}

    for task in instance:
        if task[1] >= current_time:
            task_id = task[0]
            Arrival[task_id] = task[1]
            TaskSize[task_id] = task[2]
            ProductType[task_id] = task[3]
            if policy == 'phs':
                Departure[task_id] = task[4]
            elif policy == 'dla':
                Departure[task_id] = Arrival[task_id] + product_types[ProductType[task_id]]['mean_dos']
            elif policy == 'cfa':
                Departure[task_id] = Arrival[task_id] + product_types[ProductType[task_id]]['mean_dos'] + product_types[ProductType[task_id]]['std_dos'] * theta_cfa

    LocationCapacity = {Location: 1 for Location in LocationSet}  # Assuming each location can hold one task

    #Generate distance dictionary
    Distance = {}
    TravelTime = {}

    assignment_time = 1
    retrieval_time = 1
    reallocation_time = 1

    for Location1 in LocationSet:
        for Location2 in LocationSet:
            if Location1 == Location2:
                Distance[(Location1, Location2)] = 0
                TravelTime[(Location1, Location2)] = 0
            else:
                if Location1 == 0:
                    extra_time = assignment_time
                elif Location2 == 0:
                    extra_time = retrieval_time
                else:
                    extra_time = reallocation_time

                Distance[(Location1, Location2)] = get_distance(Location1, Location2, warehouse, G, double_aisle=double_aisle, distance_type='manhattan')
                TravelTime[(Location1, Location2)] = get_distance(Location1, Location2, warehouse, G, double_aisle=double_aisle, distance_type='path') + extra_time

    MinDistance = 2
    MaxReallocations = max_reallocations

    # Helper: periods per task
    def PeriodSetTask(task):
        return [t for t in PeriodSet if Arrival[task] <= t <= Departure[task]]

    def PeriodSetMinusTask(task):
        return [t for t in PeriodSet if Arrival[task] <= t <= Departure[task] - 1]
    
    if max_reallocations > -5:

        # === Model ===
        model = gp.Model('Warehouse_Task_Assignment')

        # === Decision Variables ===
        Assignment = model.addVars(TaskSet, LocationSet, PeriodSet, vtype=GRB.BINARY, name="Assignment")
        Reallocation = model.addVars(TaskSet, LocationSet, LocationSet, PeriodSet, vtype=GRB.CONTINUOUS, name="Reallocation")
        DecisionLocationType = model.addVars(StorageLocationSet, LocationTypeSet, PeriodSet, vtype=GRB.BINARY, name="DecisionLocationType")
        ChangeLocationType = model.addVars(StorageLocationSet, LocationTypeSet, LocationTypeSet, PeriodSet, vtype=GRB.CONTINUOUS, name="ChangeLocationType")

        # === Objective ===
        obj = gp.LinExpr()

        for k in TaskSet:
            for m in StorageLocationSet:
                obj += Assignment[k, m, Arrival[k]] * TravelTime[m, In[k]]  
                obj += Assignment[k, m, Departure[k]] * TravelTime[m, Out[k]]
            for m in StorageLocationSet:
                for l in StorageLocationSet:
                    for t in PeriodSetMinusTask(k):
                        obj += Reallocation[k, m, l, t] * TravelTime[m, l]
        for m in StorageLocationSet:
            for u in LocationTypeSet:
                for w in LocationTypeSet:
                    if u != w:
                        for t in PeriodSet[:-1]:
                            obj += ChangeLocationType[m, u, w, t] * 100

        model.setObjective(obj, GRB.MINIMIZE)

        for loc, type in first_locations.items():
            type_id = next((i for i, v in location_types.items() if v['name'] == type), None)
            print('Type id:', type_id, 'Type:', type)
            model.addConstr(DecisionLocationType[loc, type_id, current_time] == 1, name=f"FirstLocation_{loc}_{type}_{t}")

        # === Constraints ===

        # Assign task exactly once per period
        for k in TaskSet:
            for t in PeriodSetTask(k):
                model.addConstr(gp.quicksum(Assignment[k, m, t] for m in StorageLocationSet) == 1,
                                name=f"AssignOnce_{k}_{t}")
        
        
        # Assign task to available locations
        for k in TaskSet:
            for m in StorageLocationSet:
                for t in PeriodSetTask(k):
                    model.addConstr(Assignment[k, m, t] <= gp.quicksum(DecisionLocationType[m, u, t] * Compatibility[ProductType[k], u] for u in LocationTypeSet),
                                    name=f"AssignAvailable_{k}_{m}_{t}")
    
        # All locations must be assigned to a location type
        for m in StorageLocationSet:
            for t in PeriodSet:
                model.addConstr(gp.quicksum(DecisionLocationType[m, u, t] for u in LocationTypeSet) == 1,
                                name=f"AssignLocationType_{m}_{t}")
    
        # Location capacity
        for m in StorageLocationSet:
            for t in PeriodSet:
                model.addConstr(
                    gp.quicksum(Assignment[k, m, t] * TaskSize[k] for k in TaskSet if t in PeriodSetTask(k)) <= LocationCapacity[m],
                    name=f"Capacity_{m}_{t}"
                )

        
        # Perigosity
        for u in LocationTypeSet:
            for w in LocationTypeSet:
                if (u, w) in IncompatibleSet:
                    for m in StorageLocationSet:
                        for l in StorageLocationSet:
                            for t in PeriodSet:
                                model.addConstr(
                                    MinDistance * (DecisionLocationType[m, u, t] + DecisionLocationType[l, w, t] - 1) <= Distance[m, l],
                                    name=f"Incompatible_{k}_{u}_{m}_{l}_{t}"
                                )
        

        # Reallocation link
        for k in TaskSet:
            for m in StorageLocationSet:
                for l in StorageLocationSet:
                    if m != l:
                        for t in PeriodSetMinusTask(k):
                            model.addConstr(
                                Assignment[k, m, t] + Assignment[k, l, t+1] - Reallocation[k, m, l, t] <= 1,
                                name=f"ReallocLink_{k}_{m}_{l}_{t}"
                            )
        
        # Location change link
        for m in StorageLocationSet:
            for u in LocationTypeSet:
                for w in LocationTypeSet:
                    if u != w:
                        for t in PeriodSet[:-1]:
                            model.addConstr(
                                DecisionLocationType[m, u, t] + DecisionLocationType[m, w, t+1] - ChangeLocationType[m, u, w, t] <= 1,
                                name=f"ChangeLink_{m}_{u}_{l}_{w}_{t}"
                            )


        # Constraint (6) - Maximum number of reallocations per time period
        for t in PeriodSet:
            model.addConstr(
                gp.quicksum(Reallocation[k, m, l, t] for k in TaskSet for m in StorageLocationSet for l in StorageLocationSet) <= MaxReallocations,
                name=f"MaxRealloc_{t}"
            )

        #Inforce that any assignment outside the task period is 0
        for k in TaskSet:
            for m in StorageLocationSet:
                for t in PeriodSet:
                    if t < Arrival[k] or t > Departure[k]:
                        model.addConstr(Assignment[k, m, t] == 0, name=f"OutsidePeriod_{k}_{m}_{t}")

        
        #Inforce all assignment variables to be the same for all periods
        for k in TaskSet:
            for m in StorageLocationSet:
                for t in PeriodSet:
                    if t > Arrival[k] and t <= Departure[k]:
                        model.addConstr(Assignment[k, m, t] == Assignment[k, m, Arrival[k]], name=f"SamePeriod_{k}_{m}_{t}")

        # Set all reallocation variables to 0
        for t in PeriodSet:
            for k in TaskSet:
                for m in StorageLocationSet:
                    for l in StorageLocationSet:
                        model.addConstr(Reallocation[k, m, l, t] == 0, name=f"Reallocation_{k}_{m}_{l}_{t}")
        
        '''
        # Inforce all location decision variables to be the same for all periods
        for m in StorageLocationSet:
            for u in LocationTypeSet:
                for t in PeriodSet[1:]:
                    model.addConstr(DecisionLocationType[m, u, t] == DecisionLocationType[m, u, t-1], name=f"SameLocationType_{m}_{u}_{t}")
        
        # Set all change location type variables to 0
        for t in PeriodSet:
            for m in StorageLocationSet:
                for u in LocationTypeSet:
                    for w in LocationTypeSet:
                        if u != w:
                            model.addConstr(ChangeLocationType[m, u, w, t] == 0, name=f"ChangeLocationType_{m}_{u}_{w}_{t}")
        '''

        # Optimize with a 5% gap
        model.Params.MIPGap = 0.01
        model.optimize()

        # === Output solution ===
        if model.status == GRB.OPTIMAL:
            print(f"Optimal objective value: {model.objVal}")
            for k in TaskSet:
                for m in LocationSet:
                    for t in PeriodSet:
                        if Assignment[k, m, t].X > 0.5:
                            print(f"Task {k} assigned to location {m} at period {t}")
        else:
            print("No optimal solution found.")

        #Get the current time solution
        currrent_time_solution = []
        for k in TaskSet:
            for m in LocationSet:
                for t in PeriodSet:
                    if Assignment[k, m, t].X > 0.5:
                        currrent_time_solution.append((k, m))
                        break

        # Suppose you have multiple time steps (say, T=10):
        warehouse_states = []
        for t in PeriodSet:
            id_vector_t = []
            product_vector_t = []
            dos_vector_t = []
            
            id_vector_t = [next((k for k in TaskSet if Assignment[k, m, t].X > 0.5), None) for m in StorageLocationSet]
            product_vector_t = [ProductType[k] if k is not None else None for k in id_vector_t]
            dos_vector_t = [Departure[k]-t+1 if k is not None else None for k in id_vector_t]
            location_type_t = [next((u for u in LocationTypeSet if DecisionLocationType[m, u, t].X > 0.5), None) for m in StorageLocationSet]

            warehouse_states.append((id_vector_t, product_vector_t, dos_vector_t, location_type_t))

    else:

        # === Model ===
        model = gp.Model('Warehouse_Task_Assignment')

        # === Decision Variables ===
        Assignment = model.addVars(TaskSet, LocationSet, vtype=GRB.BINARY, name="Assignment")
        UsedCapacity = model.addVars(StorageLocationSet, PeriodSet, vtype=GRB.CONTINUOUS, name="UsedCapacity")
        # === Objective ===
        obj = gp.LinExpr()

        for k in TaskSet:
            for m in StorageLocationSet:
                obj += Assignment[k, m] * (TravelTime[m, In[k]] + TravelTime[m, Out[k]])

        model.setObjective(obj, GRB.MINIMIZE)

        # === Constraints ===

        # Constraint (2) - Assign task exactly once per period
        for k in TaskSet:
            model.addConstr(gp.quicksum(Assignment[k, m] for m in StorageLocationSet) == 1,
                            name=f"AssignOnce_{k}")
            
        #Define the used capacity for each location
        for m in StorageLocationSet:
            for t in PeriodSet:
                model.addConstr(
                    UsedCapacity[m, t] == gp.quicksum(Assignment[k, m] * TaskSize[k] for k in TaskSet if t in PeriodSetTask(k)),
                    name=f"UsedCapacity_{m}_{t}"
                )
        # Constraint (3) - Location capacity
        for m in StorageLocationSet:
            for t in PeriodSet:
                model.addConstr(
                    UsedCapacity[m, t] <= LocationCapacity[m],
                    name=f"Capacity_{m}_{t}"
                )

        # Constraint (5) - Incompatible task distance
        for (k, u) in IncompatibleSet:
            for m in StorageLocationSet:
                for l in StorageLocationSet:
                    model.addConstr(
                        MinDistance * (Assignment[k, m] + Assignment[u, l] - 1) <= Distance[m, l],
                        name=f"Incompatible_{k}_{u}_{m}_{l}"
                    )

        # Optimize with a 5% gap
        model.Params.MIPGap = 0.30
        model.optimize()

        # === Output solution ===
        if model.status == GRB.OPTIMAL:
            print(f"Optimal objective value: {model.objVal}")
            for k in TaskSet:
                for m in LocationSet:
                    if Assignment[k, m].X > 0.5:
                        print(f"Task {k} assigned to location {m}")
        else:
            print("No optimal solution found.")

        #Get the current time solution
        currrent_time_solution = []
        for k in TaskSet:
            for m in LocationSet:
                if Assignment[k, m].X > 0.5:
                    currrent_time_solution.append((k, m))
                    break


        # Suppose you have multiple time steps (say, T=10):
        warehouse_states = []
        for t in PeriodSet:
            id_vector_t = []
            product_vector_t = []
            dos_vector_t = []
            
            id_vector_t = [next((k for k in TaskSet if Assignment[k, m].X > 0.5 and t >= Arrival[k] and t <= Departure[k]), None) for m in StorageLocationSet]
            product_vector_t = [ProductType[k] if k is not None else None for k in id_vector_t]
            dos_vector_t = [Departure[k]-t+1 if k is not None else None for k in id_vector_t]
            location_type_t = [next((u for u in LocationTypeSet if DecisionLocationType[m, u, t].X > 0.5), None) for m in StorageLocationSet]

            warehouse_states.append((id_vector_t, product_vector_t, dos_vector_t, location_type_t))

    return model.objVal, currrent_time_solution, warehouse_states



# Example usage:
number_of_blocks = 1
number_of_aisles = 4
number_of_locations_per_aisle = 5
in_out = 3
double_aisle = False

# Generate warehouse and graph
warehouse, G = generate_warehouse(number_of_blocks=number_of_blocks, number_of_aisles=number_of_aisles, number_of_locations_per_aisle=number_of_locations_per_aisle, in_out=in_out, double_aisle=double_aisle)
sections = assign_sections(warehouse, 0, G, double_aisle=False)


#Read the instances from the json file
with open('instances/instances_seed_1.json', 'r') as f:
    instance = json.load(f)

'''
product_names_vector = ["P1", "P2", "P3"]
allowed_location_types_vector = [["B1", "C2", "F3"],
                                ["C2", "F3"],
                                ["F3"]]
mean_size_vector = [1, 2, 3]
std_size_vector = [0, 0, 0]
mean_dos_vector = [1, 5, 8]
std_dos_vector = [0, 0, 0]

location_names_vector = ["B1", "C2", "F3"]
incompatible_location_types_vector = [["F3"], [], ["B1"]]

from instance_generator import generate_product_types, generate_location_types
location_types = generate_location_types(location_names_vector, incompatible_location_types_vector)
print('Location types:', location_types)
product_types = generate_product_types(location_types, product_names_vector, allowed_location_types_vector, mean_size_vector, std_size_vector, mean_dos_vector, std_dos_vector)
'''

xlsx_file = 'datacation_data.xlsx'

# Read the Excel file
df_products = pd.read_excel(xlsx_file, sheet_name='Products')
df_locations = pd.read_excel(xlsx_file, sheet_name='Locations')
df_product_location = pd.read_excel(xlsx_file, sheet_name='ProductLocation')

location_types = {}
for index, row in df_locations.iterrows():
    location_types[index] = {
        'name': row['LocationName'],
        'incompatible_location_types': []
    }

for index, row in df_locations.iterrows():
    incompatible_location_types_names = str(row['IncompatibleLocationName']).split(',')
    incompatible_location_types = [next((i for i, v in location_types.items() if v['name'] == loc), None) for loc in incompatible_location_types_names]
    location_types[index]['incompatible_location_types'] = incompatible_location_types

product_types = {}
for index, row in df_products.iterrows():
    product_types[index] = {
        'name': row['ProductName'],
        'mean_size': row['MeanSize'],
        'std_size': row['StdSize'],
        'mean_dos': row['MeanDOS'],
        'std_dos': row['StdDOS'],
        'allowed_location_types': []
    }

for index, row in df_product_location.iterrows():
    product_name = row['ProductName']
    location_id = row['LocationName']
    product_id = next((i for i, v in product_types.items() if v['name'] == product_name), None)
    location_id = next((i for i, v in location_types.items() if v['name'] == location_id), None)
    if product_id in product_types and location_id in location_types:
        product_types[product_id]['allowed_location_types'].append(location_id)

#Only consider the products that are in the instance
product_types = {k: v for k, v in product_types.items() if k in [task[3] for task in instance]}

# Only consider the locations I say
considered_location_types = ["B1", "C2", "F3"]
location_types = {k: v for k, v in location_types.items() if v['name'] in considered_location_types}


first_locations = {1: "B1", 2: "B1", 3: "B1", 4: "B1", 5: "B1", 6: "B1", 7: "B1", 8: "B1", 9: "B1", 10: "B1",
                     11: "C2", 12: "C2", 13: "C2", 14: "C2", 15: "C2", 16: "C2", 17: "C2", 18: "C2", 19: "C2", 20: "C2",
                        21: "F3", 22: "F3", 23: "F3", 24: "F3", 25: "F3"}


obj, current_time_solution, warehouse_states = lp(warehouse, G, product_types, location_types, first_locations, instance, 0, double_aisle=double_aisle, policy='phs', max_reallocations=2, theta_cfa=0.5)

print(f"\nObjective value: {obj}")

'''
print("\nCurrent time solution:")
for task, location in current_time_solution:
    print(f"Task {task} assigned to location {location}")
'''

# Save frames
frame_paths = save_warehouse_frames(warehouse, warehouse_states, G, product_types, location_types, sections)

# Create GIF
gif_path = create_gif_from_frames(frame_paths)
print(f"\n🎞️  GIF created at: {gif_path}")

