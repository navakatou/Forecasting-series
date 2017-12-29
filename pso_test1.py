import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to test the algorithm Particle Swarm Optimization
def objective_function(vector):
	result = np.sum([x**2 for x in vector])
	return result

# Function to make a random vector based on the limits
def random_vector(minmax):
	vector = []
	for i in range(2):
		tem = minmax[0] + (minmax[1]-minmax[0]*np.random.random())
		vector.append(tem)
	return vector

# Function to create every particle that will be used in PSO
def create_particle(search_space, vel_space):
	particle = {}
	particle['position'] = random_vector(search_space)
	particle['cost'] = objective_function(particle['position'])
	particle['b_position'] = particle['position']
	particle['b_cost'] = particle['cost']
	particle['velocity'] = random_vector(vel_space)
	return particle
 
# Function to find the best element from all the particles, that will be used to update them
def get_global_best(population, current_best = None):
	#particles_cost = sorted([x.get('cost') for x in population], reverse = True)
	particles_cost = sorted([x.get('cost') for x in population])
	best = [d for d in population if d['cost']==particles_cost[0]]
	best = dict(best[0])
	#print current_best
	#best = dict(filter(lambda x: x.get('cost') == particles_cost[0],population))
	if (current_best == None or (best['cost'] <= current_best['cost'])):
		current_best = {}
		current_best['position'] = best['position'] 
		current_best['cost'] = best['cost']
	#print 'Calculating the best particle'
	return current_best

# Function to find the best position of the particle
def update_best_position(particle):
	if (particle['cost'] > particle['b_cost']):
		particle['b_cost'] = particle['cost']
  		particle['b_position'] = particle['position']

  	return particle

# Function to update the velocity of a particle based on the best particle
def update_velocity(particle, gbest, max_v, c1, c2):
	v1 = []
	v2 = []
	for i in range(len(particle['velocity'])):
		v1.append(c1 * np.random.random() * (particle['b_position'][i]-particle['position'][i]))
		v2.append(c2 * np.random.random() * (gbest['position'][i]-particle['position'][i]))
	tmp =  zip(v1, v2, particle['velocity'])
	var = [sum(item) for item in tmp]
		
	if particle['velocity'][0] > max_v:
		particle['velocity'][0] = max_v 
	if particle['velocity'][1] > max_v:
		particle['velocity'][1] = max_v 
	if particle['velocity'][0] < -max_v:
		particle['velocity'][0] = -max_v 
	if particle['velocity'][1] < -max_v:
		particle['velocity'][1] = -max_v 
	
	return particle

# Function to update the postion of a particle based on his velocity. This function also checks if 
# it is  between the limits 
def update_position(particl, bounds):
	tm = [(i+j) for i,j in zip(particl['position'], particl['velocity'])]
	particl['position'] = tm
	for k in range(len(particl['position'])):
		if particl['position'][k] > bounds[1]:
			particl['position'][k]=bounds[1]-abs((particl['position'][k]-bounds[1]))
			particl['velocity'][k] *= -1.0
		if particl['position'][k] < bounds[0]:
			particl['position'][k]=bounds[0]+abs((particl['position'][k]-bounds[0]))
			particl['velocity'][k] *= -1.0

	return particl

# Function to run all the algotrithm with PSO
def search(max_gens, search_s, vel_s, pop_size, max_vel, c1, c2):
	popult = [ create_particle(search_s,vel_s) for x in range(pop_size)]
	gbest = get_global_best(popult)
	for i in range(max_gens):
		for item in popult:
			item = update_velocity(item, gbest, max_vel, c1, c2)
			item = update_position(item, search_s)
			item['cost'] = objective_function(item['position'])
			item = update_best_position(item)
		gbest = get_global_best(popult, gbest)
		print 'gen {}, fitness = {}'.format(i+1, gbest['cost'])

	return gbest


if __name__ == "__main__":

	pop_size = 50 # Number of particles
	search_s = [-5, 5] # Limits of position of a square region
	vel_s = [-1,1] # Limits of velocity of a square region
	max_gens = 100 # Number of iterations
	max_vel = 100.0 # Maximum velocity of a particle
	c1, c2 = 2.0, 2.0 # Constants to update the velocity
	# execute the algorithm
	best = search(max_gens, search_s, vel_s, pop_size, max_vel, c1, c2)
	print 'done! Solution: f = {}, s = {}'.format(best['cost'], best['position'])


