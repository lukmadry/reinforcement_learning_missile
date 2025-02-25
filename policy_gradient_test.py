import numpy as np
import matplotlib.pyplot as plt
import math

# we start at x, y = (0, 0) and want to shoot a target that sits at 10

GRAVITY = 9.81

# model here is REWARD(a_1, a_2, a_3) = \ln 2 + a_1 \ln v_x + a_2 \ln v_y - a_3 \ln GRAVITY
# and we do not know a_1, a_2, a_3 (they are supposed to be all equal to one)
# for the model with resistance - we will learn the Taylor coefficients of the approximation (in theory they should coverge to the same thing)

def compute_endpoint(init_speed_x, init_speed_y):
	# y(t) = v_y t - g t^2/2
	# which means that hitting time of zero can be computed to be t (v_y - gt/2) = 0
	# so v_y = gt/2 => t = 2v_y/g
	# and the position after that time is x(t) = v_x t
	return init_speed_x * 2 * init_speed_y / GRAVITY

def my_sign(x):
	if (x>=0):
		return 1
	return -1

def compute_endpoint_numerically_res(init_speed_x, init_speed_y, air_res, N, timestep):
	n = 0
	posx, posy = 0, 0
	velx, vely = init_speed_x, init_speed_y
	while (n < N and posy >= 0):
		velx -= air_res * (velx) * (velx) * timestep
		vely += -GRAVITY * timestep - my_sign(vely) * air_res * (vely) * (vely) * timestep

		posx += velx * timestep
		posy += vely * timestep
		n += 1
	return posx, posy, n

def compute_approximation(init_speed_x, init_speed_y, taylor_step, coefficients):
	finval = 0
	i = 0
	assert(len(coefficients) == sum(range(2,taylor_step+1)))
	for k in range(1,taylor_step):
		# this means that we will do k sums, \sum_{i_1}^2 \sum_{i_2}^2 ... \sum_{i_k}^2 ()
		kth_step = 0
		for j in range(k+1):
			loc = coefficients[i] * math.comb(k, j) * math.pow(init_speed_x, j) * math.pow(init_speed_y, k-j)
			i += 1
			kth_step += loc
		finval += kth_step
	return finval

def create_approx_vector(init_speed_x, init_speed_y, taylor_step):
	vec = []
	for k in range(1,taylor_step+1):
		for j in range(k+1):
			vec.append( math.comb(k, j) * math.pow(init_speed_x, j) * math.pow(init_speed_y, k-j) )
	return np.array(vec)

def compute_reward(init_speed_x, init_speed_y, coefs):
	return np.log(2) + coefs[0] * np.log(init_speed_x) + coefs[1] * np.log(init_speed_y) - coefs[2] * GRAVITY

def compute_update(init_speed_x, init_speed_y, proposed_coefs):
	true_reward = compute_reward(init_speed_x, init_speed_y, [1, 1, 1])
	est_reward = compute_reward(init_speed_x, init_speed_y, proposed_coefs)
	grad = np.array([np.log(init_speed_x), np.log(init_speed_y), - GRAVITY])
	return (true_reward - est_reward) * grad

def compute_update_taylor_upd(init_speed_x, init_speed_y, proposed_coefs, taylor_step, simulator_kwargs=None, with_air_res=True, verbose=True):
	if with_air_res:
		true_posx, true_posy, _ = compute_endpoint_numerically_res(init_speed_x, init_speed_y, **simulator_kwargs)
	else:
		true_posx = compute_endpoint(init_speed_x, init_speed_y)
	approx_vec = create_approx_vector(init_speed_x, init_speed_y, taylor_step)
	if (verbose):
		print('approximation vector', approx_vec)
	approx_val = np.dot( approx_vec, proposed_coefs )
	if verbose:
		print('true val', true_posx)
		print('approx val', approx_val)
	return (true_posx - approx_val) * approx_vec, (true_posx - approx_val)

def make_updates(N, eps=1e-1, sd=4, learning_rate=1e-4, taylor_step=3, grad_cap=10e2, verbosity_threshold=100):
	grad_stop_cond = 8
	n = 1
	coefs = np.random.normal(0, scale=sd/100, size=sum(range(2,taylor_step+2)))
	grad_norms = []
	errors = []
	kwargs = {'air_res': 0.05, 'N': 1000, 'timestep': 1e-3}

	while (grad_stop_cond > eps and n < N):

		speeds = np.random.uniform(0, 10, size=2)
		upd, error = compute_update_taylor_upd(speeds[0], speeds[1], coefs, taylor_step=taylor_step, with_air_res=True, 
			simulator_kwargs=kwargs, verbose=(n%verbosity_threshold==0))
		upd = np.minimum(grad_cap, upd)
		errors.append(math.fabs(error))

		if (n % verbosity_threshold == 0):
			print('upd', upd, '\n')
			print('errors', np.mean(errors[-verbosity_threshold:]))

		coefs += learning_rate * upd
		grad_norm = np.linalg.norm(upd)
		grad_norms.append(grad_norm)

		if (n > 1000):
			grad_stop_cond = np.mean(grad_norms[-1000:])
		if (n% verbosity_threshold == 0):
			print('grad', grad_stop_cond)
		n += 1

	return (coefs, grad_stop_cond, n)

print(make_updates(100000, verbosity_threshold=10000, eps=1e-4, taylor_step=2, learning_rate=1e-5))