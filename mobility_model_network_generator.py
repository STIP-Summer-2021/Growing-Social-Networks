import os, random, json
import numpy as np
import pandas as pd
from datetime import datetime
from calendar import monthrange
import matplotlib.pyplot as plt
cwd = os.path.dirname(os.path.abspath(__file__)) # reference to the path of this file

class Constants:
    '''
    Class that contains references to different constant values. 
    '''
    SUSCEPTIBLE = 'S'
    PRE_SYMPTOMATIC = 'Ip'
    ASYMPTOMATIC = 'Ia'
    SYMPTOMATIC = 'Is'
    EXPOSED = 'E'
    RECOVERED = 'R'
    BUCKET_LOOKUP = {arrival_time:bucket for bucket, (start, stop) in enumerate([(0,1), (1,4), (4,8), (8, 16), (16, 96)]) for arrival_time in range(start, stop)}
    HOUSEHOLD_IDS = [str(i) for i in range(1,8)]
    TICKS = range(96)

class NAICS:
    '''
    This is a container class for each NAICS code. Conatains the distribution of arrival times for 
    each category of POI. This class is intiated inside of the model class using the create_naics method.
    '''
    def __init__(self, dist):
        self.dist = dist
        self.ticks = range(96)

    def get_times(self, n):
        # Returns n samples from the distribution
        return np.random.choice(self.ticks, n, p=self.dist)

    def get_time(self):
        # Returns 1 sample from the arrival dist
        return np.random.choice(self.ticks, p=self.dist)
    
    def plot_samples(self, n=100_000):
        # Method for plotting the distribution of arrival times
        ticks = {i:0 for i in range(96)}
        for sample in self.get_times(n):
            ticks[sample] += 1
        
        plt.bar(ticks.keys(), ticks.values())
        plt.xticks(range(0,96,4), labels=[(i*15)/60 for i in range(0,96,4)])
        plt.show()

class Place:
    '''
    Parent class for POIs and Home objects. Contains a reference to the model class as well as a dictionary of agents at the location.
    '''
    def __init__(self, main_model):
        self.agents = [] #{Constants.SUSCEPTIBLE:[], Constants.ASYMPTOMATIC:[], Constants.SYMPTOMATIC:[], Constants.PRE_SYMPTOMATIC:[], Constants.EXPOSED:[], Constants.RECOVERED:[]}
        self.model = main_model
    
    def reset_agent_dict(self):
        self.agents = []#{Constants.SUSCEPTIBLE:[], Constants.ASYMPTOMATIC:[], Constants.SYMPTOMATIC:[], Constants.PRE_SYMPTOMATIC:[], Constants.EXPOSED:[], Constants.RECOVERED:[]}

class POI(Place):
    '''
    This class is used to create the POIs agents can visit. 
    Each POI is created using the parsed CSV "{year}_Fairfax_POI_Data.csv", derived from SafeGraph data.
    Intialized in the Model class during init from the create_pois method.

    Contains the following information:

    self.place_id = SafeGraph Placekey
    self.place_name = Placename assigned by SafeGraph
    self.lat = POI latitude
    self.lon = POI longitude
    
    self.cbg = The fips code of the CBG this POI resides in
    self.naics_code = NAICS code of the business, essentially categories given by the government for businesses


    self.dwell_dist = A list of possible dwell times, generated from the SafeGraph bucketed dwell times
    self.agents = A list of agents that are currently at this POI
    '''
    def __init__(self, place_id ,row_data, main_model):
        super().__init__(main_model)
        self.place_id = place_id
        self.visitor_cbg = {}
        self.agent_logging = set()
        self.tot_visits = 0

        self.hour_visits = np.zeros(24)
        self.hourly_logger = {i:set() for i in range(24)}

        self.naics_code = row_data['naics_code'][0:2] 
        
        
        self.dwell_logger = {bucket:0 for bucket in range(5)}

        self.arrival_dist = np.array([val for val in eval(row_data['popularity_by_hour']) for _ in range(4)])
        self.arrival_dist = self.arrival_dist / np.sum(self.arrival_dist)
        
        self.generate_dwell_dist(row_data['bucketed_dwell_times'])

    def reset_hour_logger(self):
        for hour, agents in self.hourly_logger.items():
            self.hour_visits[hour] += len(agents)
        
         
        self.hourly_logger = {i:set() for i in range(24)}

    def generate_dwell_dist(self, bucketed_dict):
        # Use the SafeGraph bucketed dwell times to create a dwell distribution 
        lookup = {'<5': (0,1), '5-10': (0,1), '11-20': (0,1), '21-60': (1,4), '61-120': (4,8), '121-240': (8, 16), '>240': (16, 96)} # time converted to ticks 
        dist = []
        for key, val in eval(bucketed_dict).items(): # loop through each key:val in the bucketed dwell dict and lookup the associated time in ticks
            if key.startswith('>'): 
                dist.extend(list((np.random.gamma(1,3.5,val).astype(int) + 16))) # <== this probably needs to be revisited, uses a gamma distribution for times greater than 240 minutes
            else:    
                low, high = lookup[key]
                dist.extend(list((np.random.randint(low, high, val)).astype(int))) # generate a uniform distribution of values based on the lookup table
        
        self.dwell_dist = dist # store the distribution

    def get_agent_dwell_time(self, use_data=True): # returns a likely dwell time from the POIs dwell dist  
        if use_data:
            return self.dwell_dist[np.random.randint(0, len(self.dwell_dist))]
        else:
            return np.random.randint(0, 95) # randomly up to the length of the day

    def get_most_likely_time(self, use_data=True, use_own_dist=False):
        if use_data:
            if use_own_dist:
                return np.random.choice(Constants.TICKS,  p=self.arrival_dist)
            else:
                return self.model.naics[self.naics_code].get_time()
        else:
            return np.random.randint(0, 95) # randomly through the entire day



class CBG:
    '''
    Class for representing each CBG contained inside of Fairfax County (or really whatever shapefile is passed to it)
    During the init phase of each CBG, the agents and household are created and stored in the model class


    Conatins the following information:
    self.cbg_fips = 12 digit FIPS code for each CBG 
    self.household_sizes = A numpy array containing the number of each size household inside the CBG
    self.census_pop = This is the theoretical population of the CBG as collected by the census
    self.propensity_to_leave = The percentage of the CBG that doesn't leave home during the day. Derived from SafeGraph data.

    self.households = List of household objects inside each CBG
    self.agents = List of agents from the CBG
    '''
    def __init__(self, cbg_fips, row_data, main_model):
        self.model = main_model 
        self.cbg_fips = cbg_fips
        self.household_sizes = row_data[Constants.HOUSEHOLD_IDS].values.astype(int) # get the household size array from the row data
        # self.census_pop = row_data['total_pop']
        self.propensity_to_leave = row_data['Perc_Stay_Home']
        self.topic_probs = eval(row_data['CBG_Topic_Prob'])
        self.num_phones = row_data['number_devices_residing']
        self.observed_visits = row_data['total_visits']

        self.households = []
        self.agents = []
        self.create_households() 
        
    def create_households(self):
        for house_size, num_houses in enumerate(self.household_sizes):
            for _ in range(num_houses):
                self.households.append(Household(len(self.households), house_size+1, self, self.model))

class Household(Place):
    '''
    Class representing agent households. Initiated with a household size and parent CBG, creates N agents
    based on household size. 

    Information contained:
    self.agents = List of agents that "live" at this house 
    self.home_id = ID of the home object in the CBG households list
    self.cbg = Reference to the parent CBG object
    self.size = Household size

    '''
    def __init__(self, idx, size, parent_cbg, main_model):
        super().__init__(main_model)
        self.home_id = idx
        self.cbg = parent_cbg
        self.size = int(size)
        self.model.households.append(self)

        self.create_agents()
    
    def create_agents(self):
        # Creates N agents based on household size. A reference to the agent is stored in the house and parent cbg
        for _ in range(self.size):
            
            if self.model.use_LDA:
                topic = np.random.choice(range(len(self.model.poi_topics)), p=self.cbg.topic_probs) 
            elif self.model.use_prob:
                topic = self.cbg.cbg_fips
            else:
                topic = 0
                
            agent = Agent(home_obj=self, main_model=self.model, topic=topic)

            self.cbg.agents.append(agent) 


class Agent:
    '''
    Class representing individuals for the Model class. 

    Information contained:
    self.id = Unique ID for the agent, helps for checking the interactions between agents 
    
    self.current_loc = Reference to the current location of the agent
    self.home_obj = Reference to the home object of the agent

    self.infection_status = The current infection status of the agent 

    self.current_interactions = A set to check for interactions 
    self.current_infections = The number of infections that an agent had per tick

    self.total_interactions = The total number of interactions the agent had
    self.total_infections = The total number of agents that this agent infected

    self.schedule = List containing references to POIs the agent will go to     

    self.incubation_time = Agents undergo an incubation period before they reach either the symptomatic or asymptomatic phase
    self.infection_time = Amount of time that an agent spends in either the symptomatic or asymptomatic phases

    '''
    total = 0 # counter to create agent ID 
    def __init__(self, home_obj, main_model, infection_status=Constants.SUSCEPTIBLE, topic=0):
        self.id = Agent.total # unique ID
        self.model = main_model
        self.friends = list()
        self.topic = topic
        # AGENTS START AT HOME
        self.current_loc = home_obj
        self.home_obj = home_obj
        self.home_fips = home_obj.cbg.cbg_fips
        
        # AGENTS START AS SUSCEPTIBLE 'S'
        self.infection_status = infection_status 
        self.lifetime_trips = 0

        self.model.agents.append(self)
        Agent.total += 1
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # MOVEMENT LOGIC

    def check_range(self, low, high, guess): # simple function to check if a number is in between some variables
        return (guess >= high) or (guess <= low)

    def get_locs(self, n, with_replacement=False):
        '''
        I added the parameter "with_replacement" because I believe that since we are simulating the number of observed unique visits, this gives us 
        the best chance of reproducing the observed data. It may be cheating, but the jaccard score also went down when I sampled with replacement. 
        '''

        if self.model.use_LDA and not self.model.use_prob: # LDA
            if not with_replacement: # if we use no replacement you can only have up to the number of non-zero probs 
                n = n if n < self.model.poi_topics[self.topic]['non_zeros'] else self.model.poi_topics[self.topic]['non_zeros']
            
            locs = np.random.choice(self.model.poi_topics[self.topic]['ids'], n, p=self.model.poi_topics[self.topic]['probs'], replace=with_replacement) # select some random POIs
        

        elif not self.model.use_LDA and self.model.use_prob: # Prob Model
            if not with_replacement:  
                n = n if n < self.model.poi_topics[self.topic]['non_zeros'] else self.model.poi_topics[self.topic]['non_zeros']

            locs = np.random.choice(self.model.poi_topics[self.topic]['ids'], n, p=self.model.poi_topics[self.topic]['probs'], replace=with_replacement) 
        

        elif not self.model.use_LDA and not self.model.use_prob: # Random Model
            locs = np.random.choice(self.model.poi_topics[self.topic]['ids'], n, replace=with_replacement)
        
        return locs


    def build_schedule(self, num_locations):
        # Daily schedule builder for agents. 

        temp_schdeule = list(np.zeros(96)) # 0's in the schedule represent the home object for each agent 
        time_away = 0
        self.num_visits = 0
        # if the agent is visiting places and rolls sucessfully for leaving their home
        if num_locations and np.random.random() > self.home_obj.cbg.propensity_to_leave:
            if num_locations > len(self.model.poi_topics[self.topic]) and (self.model.use_LDA or self.model.use_prob): # sometimes the prob model will have no observed trips for some cbgs so catch it here
                if not len(self.model.poi_topics[self.topic]):
                    self.schedule = temp_schdeule
                    return
                num_locations = len(self.model.poi_topics[self.topic]) 
            

            times = [] # holding var for accepted times

            # UPDATED METHOD FOR SELCTING LOCATIONS 
            # instead of selecting here, I wrote a quick method that uses different ways of selecting the locations that an agent goes to 
            # you can change those parameters either in this function call or just by changing the default parameters of that method. 
            for loc in self.get_locs(num_locations): # go through each selected POI 
                loc = self.model.pois[loc] # lookup the POI
    
                start = loc.get_most_likely_time() # pull a start time
                time = loc.get_agent_dwell_time() + 1 # slicing needs an extra index, doesn't change real dwell time

                end = start + time # pull a dwell length 
                end = 95 if end > 95 else end # if the end time goes outside of the day set it to the end 
                
                _stop = False

                for s,e in times:
                    if not all((self.check_range(s, e, start), self.check_range(s, e, end), self.check_range(start, end, s), self.check_range(start, end, e))):
                        _stop = True
                        break 

                if _stop: continue    
                
                # logging info 
                # loc.dwell_logger[Constants.BUCKET_LOOKUP[time-1]] += 1 # log the dwell time associated with the visit (time - 1) is for looking up the actual dwell time
                # loc.visitor_cbg[self.home_fips] = loc.visitor_cbg.get(self.home_fips, 0) + 1 # repeat counts
                # for tick in range(start, end):
                #     loc.hourly_logger[tick // 4].add(self)

                # loc.agent_logging.add(self)
                # loc.tot_visits += 1


                times.append((start, end)) # success!!
                temp_schdeule[start:end] = [loc for _ in range(end-start)] # assign the agent a place
                time_away += time

            self.num_visits = len(times)

        self.lifetime_trips += self.num_visits
        self.schedule = temp_schdeule
        self.time_away = time_away

    def step(self, tick):
        '''
        Moves the agent to the next location in it's schedule based on the current tick. 
        '''
        next_loc = self.schedule[tick]
        next_loc = next_loc if next_loc else self.home_obj

        next_loc.agents.append(self)  # move to next location
        self.current_loc = next_loc  # store the next location as current location
        self.check_for_friends()
     # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

    def check_for_friends(self):       
        #check to see if they become friends
        if (np.random.random() < .5) and (len(self.current_loc.agents) > 1):
            nf = random.choice(self.current_loc.agents)
            if nf in self.friends or nf == self:
                return
            # increase the count, this is so we don't have to use the sparse matrix 
            self.model.total_edges += 2
            # store agent pointers 
            nf.friends.append(self)
            self.friends.append(nf)

class Model:
    '''
    Main model class for the simulation. 

    The following describes the initialization phase. 

    1. The model landscape is initialized and all the CBGs are read in.
        
        - When a CBG object is created, it initializes a population of agents equivalent to the number of households recorded in census data.
    
    2. A subset of agents are selected to be initially infected.
        
        - A CBG can be selected as a seed for the initial infection or can be randomly distributed across the simulation.
        
        - The logic for this will likely need to be redone as currently only a proportion of a population is used. This means that if a random 
          selection of agents are selected, the number of infected agents will be higher. For example, if 5% of agents are selected for a CBG 
          of 1000 agents, there will be 100 agents infected. If 5% are chosen with a random selection the model may have 500,000 agents that 
          would mean that there are 5000 agents infected. 
    
    3. POIs are read in and created. 
 
    4. NAICS codes are read in and created. 

    year = model year
    month = model month
    sim_year = what year you want to simulate
    sim_month = what month you want to simulate


    one of these must be true, if not a random mobility model will be used. 
    use_lda - uses the lda model with the weights 
    use_prob - uses the prob model to choose location 
    ''' 

    def __init__(self, save_name='', use_LDA=False, use_prob=False, year=2019, month=1, sim_year=2019, sim_month=1, k=4, input_cbg_file='scaled_houses_100k', print_info=False):
        assert not (use_LDA and use_prob), 'Cannot use both the LDA Model and Prob Model'
        Agent.total = 0
        self.agents = []
        self.all_pois = []
        self.households = []
        
        self.poi_topics = {}
        self.pois = {}
        self.cbgs = {}
        self.naics = {}
        
        self.k = k
        self.current_tick = 0
        self.total_edges = 0
        
        self.use_LDA = use_LDA
        self.use_prob = use_prob

        self.year = year
        self.month = month
        self.sim_year = sim_year 
        self.sim_month = sim_month

        self.mobility_model_type = ''
        self.print_info = print_info
        self.cbg_file = input_cbg_file
        self.save_name = save_name
        

        self.days = monthrange(sim_year, sim_month)[1]

        self.create_pois()
        self.create_cbgs()
        self.create_naics()
        # self.create_friend_matrix()

        self.interactable_places = self.all_pois + self.households 

    # def create_friend_matrix(self):
    #     total_agents = len(self.agents)
    #     self.friend_matrix = sparse.lil_matrix((total_agents, total_agents))

    def create_pois(self):
        # read the poi data that we ran the LDA on
        poi_data = pd.read_csv(os.path.join(cwd, 'Data', 'Monthly_POIs', f'{self.year}_{self.month:02d}.csv'), dtype={'poi_cbg':str, 'naics_code':str}, index_col=0) # Read in POIs
        
        for poi, row in poi_data.iterrows(): # store all the POIs in a dictionary 
            to_add = POI(poi, row, self)
            self.pois[poi] = to_add
            self.all_pois.append(to_add)

        # if we want to use the LDA load the topic json
        if self.use_LDA and not self.use_prob:
            with open(os.path.join(cwd, 'Data', 'Topic_JSONs', f'{self.year}_{self.month:02d}.json'), 'r') as f:
                for topic, vals in json.load(f).items():
                    self.poi_topics[int(topic)] = {key:np.array(val) for key, val in vals.items()}

                    # count the non zeros just in case we use non replacement scheduling
                    self.poi_topics[int(topic)]['non_zeros'] = np.count_nonzero(vals['probs'])
                if self.print_info: print(f'Using LDA with {len(self.poi_topics)} topics.')
                self.mobility_model_type = f'lda_{len(self.poi_topics)}'


        elif not self.use_LDA and self.use_prob:
            # prob model stores poi probs under the cbg key 
            with open(os.path.join(cwd, 'Data', 'CBG_POI_Probs', f'{self.year}_{self.month:02d}.txt'), 'r') as file:
                for line in file:
                    cbg, poi_dist = line.split('_')
                    poi_dist = eval(poi_dist)
                    self.poi_topics[str(cbg)] = {'ids':np.array(list(poi_dist.keys())), 'probs':np.array(list(poi_dist.values())), 'non_zeros': np.count_nonzero(list(poi_dist.values()))}
                if self.print_info: print('Using Prob model.')
                self.mobility_model_type = 'prob'

        else:
            # random model just stores all the ids in topic "0"
            self.poi_topics = {0:{'ids':np.array(list(self.pois.keys()))}}
            if self.print_info: print(f'Using random model.')
            self.mobility_model_type = 'random'


    def create_naics(self):
        with open(os.path.join(cwd, 'Data', 'Monthly_NAICs', f'{self.year}_{self.month:02d}.txt'), 'r') as file:
            for line in file:
                naics_code, dist = line.split('_')
                dist = eval(dist, {'nan':np.nan})
                dist = dist if not np.nan in dist else None
                self.naics[naics_code] = NAICS(dist) 

    def create_cbgs(self):
        # read the raw cbg data 
        cbg_data = pd.read_csv(os.path.join(cwd, 'Data', f'{self.cbg_file}.csv'), dtype={'census_block_group':str}).set_index('census_block_group')

        # load the LDA topic dist from the target year and month 
        cbg_topic_dist = pd.read_csv(os.path.join(cwd, 'Data', 'Monthly_CBG_Topic_Dist', f'{self.year}_{self.month:02d}.csv'), dtype={'census_block_group':str}).set_index('census_block_group')
        
        # get the number of cell phones seen in the cbg during the simulated month and seed the population with that number per cbg
        cbg_panel_data = pd.read_csv(os.path.join(cwd, 'Data', 'CBG_Visits', f'{self.sim_year}_{self.sim_month:02d}.csv'), dtype={'census_block_group':str}).set_index('census_block_group')
        
        cbg_data = pd.concat([cbg_data, cbg_topic_dist, cbg_panel_data], axis=1).fillna(0)

        for cbg, row in cbg_data.iterrows():
            if (cbg not in self.poi_topics) and self.use_prob: continue #sometimes there are no observed visits from a CBG so don't make it
            self.cbgs[cbg] = CBG(cbg, row, self)


    def step(self):
        np.random.shuffle(self.agents)  # for random agent activation
        tick = self.current_tick % 96

        if not tick:  # end of the day
            # this logic was moved to build schedules
            num_visits = np.round(np.random.lognormal(1, .5, len(self.agents))).astype(int)  # generated from the paper distribution
            for agent, visits in zip(self.agents, num_visits):
                agent.build_schedule(visits)

        for agent in self.agents:  # step agents to next location in their schedule
            agent.step(tick)

        for place in self.interactable_places:  # loop through list of homes and pois to interact agents
            place.reset_agent_dict()  # then reset that places agent dict

        self.current_tick += 1
        
    def run(self):
        avg_degree = 0 
        avg_degree_time = []
        while avg_degree <= self.k:
            s = datetime.now()
            self.step()
            avg_degree = self.total_edges / len(self.agents)
            avg_degree_time.append(avg_degree)
            if self.print_info: print(f'Day: {self.current_tick//96}\tTick: {self.current_tick}\tDeg: {avg_degree}\tElapsed: {datetime.now() - s}')
        
        with open(r'Generated_Edges\k={}_n={}.csv'.format(self.k, len(self.agents)), 'w+') as out:
            print('Source,Target', file=out)
            for agent in self.agents:
                for friend in agent.friends:
                    print(agent.id, friend.id, sep=',', file=out)
        
        # pd.DataFrame(avg_degree_time, columns=['avg_deg']).to_csv(r'Outputs\Degree_Over_Time\k={}_n={}.csv'.format(self.k, len(self.agents)), index_label='tick')

