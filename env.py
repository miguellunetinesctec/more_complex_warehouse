import numpy as np
from gymnasium import Env, spaces
import copy
from miscelaneous import generate_warehouse, get_distance, generate_products, assign_sections

class WarehouseEnv(Env):

    def __init__ (self,
                  warehouse,
                  G,
                  product_types,
                  source = "external",
                  action_type = 'location',
                  time_horizon = 10,
                  seed = 20):
        
        self.action_type = action_type
        self.source = source
        self.source_file = "instances/instances.json"

        self.num_locations = warehouse['num_locations']
        self.num_sectors = 3

        self.distances_manhattan = warehouse['distances_manhattan']
        self.distances_path = warehouse['distances_path']
        self.dict_sector_capacity = warehouse['dict_sector_capacity']
        self.dict_capacity_sector = warehouse['dict_capacity_sector']

        self.product_ids = [product['id'] for product in products]
        self.incompatible_products = incompatible_products

        self.current_time = 0
        self.time_horizon = time_horizon
        if self.source == 'external':
            #Read the instances from the json file
            pass
        elif self.source == 'internal':
            self.new_tasks = [1]
            self.current_task = [1]

        self.location_task_id = [None for i in range(self.num_locations)]
        self.location_product_id = [None for i in range(self.num_locations)]
        self.location_dos = [None for i in range(self.num_locations)]
        self.location_capacity = [1 for i in range(self.num_locations)]
        self.sector_max_capacity = [sum(self.dict_sector_capacity[sector]) for sector in range(self.num_sectors)]
        self.sector_capacity = [self.sector_max_capacity[sector] for sector in range(self.num_sectors)]

        self.seed = seed
        np.random.seed(self.seed)
        
        self.obs = self.reset()

        obs_dim = len(self.reset()[0])
        low_dim = np.array([0 for i in range(obs_dim)])
        high_dim = np.array([1 for i in range(obs_dim)])
        self.observation_space = spaces.Box(low = low_dim, high = high_dim, dtype = np.float32)
        self.action_space = spaces.Discrete(self.num_locations)
    
    def update_seed(self, seed):
        self.seed = seed
        self.refresh_seed()

    def refresh_seed(self):
        np.random.seed(self.seed)

    def action_masks(self) -> list[bool]:

        mask = []

        if self.action_type == 'location':
            for loc in range(self.num_locations):
                if self.location_capacity[loc] > 0:
                    mask.append(True)
                else:
                    mask.append(False)
            
        elif self.action_type == 'sector':
            for sector in range(self.num_sectors):
                if self.sector_capacity[sector] > 0:
                    mask.append(True)
                else:
                    mask.append(False)

        return mask

    def step(self, action):

        if self.action_type == 'location':
            location_id = action
        elif self.action_type == 'sector':
            location_id = self.sector_capacity.index(max(self.sector_capacity))

        self.location_task_id[location_id] = self.current_task
        self.location_product_id[location_id] = self.product_ids[self.current_task]
        self.location_dos[location_id] = 0
        self.location_capacity[location_id] = 0
        self.sector_capacity[location_id] -= 1
        
        #Take action
        done = False
        info = {}
        reward = 2 * self.distances_path[location_id][0]

        #Update products
        self.new_tasks.pop(0)

        if self.new_tasks == []:
            if self.current_time == self.time_horizon:
                self.history_task_id.append(self.location_task_id)
                self.history_product_id.append(self.location_product_id)
                self.history_dos.append(self.location_dos)
                done = True
            else:
                self.new_tasks = []
                self.current_task = self.new_tasks[0]
                self.history_task_id.append(self.location_task_id)
                self.history_product_id.append(self.location_product_id)
                self.history_dos.append(self.location_dos)
                self.current_time += 1
        else:
            self.current_task = self.new_tasks[0]

        #Update location
        for loc in range(self.num_locations):
            self.location_capacity[loc] = self.location_capacity[loc] - 1
            if self.location_capacity[loc] == 0:
                self.location_task_id[loc] = None
                self.location_product_id[loc] = None
                self.location_dos[loc] = None
                self.location_capacity[loc] = 1

        
        obs = {'current_time': self.current_time, 'new_tasks': self.new_tasks, 'current_task': self.current_task, 'location_task_id': self.location_task_id, 'location_product_id': self.location_product_id,
                'location_dos': self.location_dos, 'location_capacity': self.location_capacity, 'current_time': self.current_time}
        
        info['next_state'] = copy.deepcopy(obs)

        self.obs = copy.deepcopy(obs)
        
        state = self.get_state(self.obs)

        return state, reward, done, done, info
    
    def get_state(self, state):
    
        data_rows = []

        if self.action_type == 'location':

            for loc in range(self.num_locations):
                data_rows.append(self.location_capacity[loc])
            
            for loc in range(self.num_locations):
                value = min(data_rows.append(self.location_dos[loc]/10),1)
                data_rows.append(value)

            for prod in range(len(self.product_ids)):
                for loc in range(self.num_locations):
                    if self.location_product_id[loc] == self.product_ids[prod]:
                        data_rows.append(1)
                    else:
                        data_rows.append(0)
        
        elif self.action_type == 'sector':
        
            for i in range(self.num_sectors):
                data_rows.append(state['capacity'][i]/state['static_info']['max_capacity'][i])
        
        return np.array(np.nan_to_num(data_rows, nan=0), dtype=np.float32)
        
        
    def reset(self, seed=0):

        self.current_time = 0
        self.new_tasks = []
        self.current_task = 0

        self.location_task_id = [None for i in range(self.num_locations)]
        self.location_product_id = [None for i in range(self.num_locations)]
        self.location_dos = [None for i in range(self.num_locations)]
        self.location_capacity = [None for i in range(self.num_locations)]

        self.history_task_id = []
        self.history_product_id = []
        self.history_dos = []

        obs = {'current_time': self.current_time, 'new_tasks': self.new_tasks, 'current_task': self.current_task, 'location_task_id': self.location_task_id, 'location_product_id': self.location_product_id,
                'location_dos': self.location_dos, 'location_capacity': self.location_capacity, 'current_time': self.current_time}
        
        self.obs = obs
        
        state = self.get_state(self.obs)

        return state, {}

    def render(self, mode='human'):
        pass

    
from stable_baselines3.common.env_checker import check_env; check_env(WarehouseEnv())
check_env(WarehouseEnv())
print('Environment is valid')


#Run the env with a random policy

'''
done = False
final_reward = 0
env = WarehouseEnv()
state,_ = env.reset()
env.render()

while not done:
    action = env.action_space.sample()
    state, reward, done, done, info = env.step(action)
    final_reward += reward
    env.render()
print(f'Final reward: {final_reward}')
#Run the env with a trained policy
'''

