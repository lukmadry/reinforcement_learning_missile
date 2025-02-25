this repoistory contains a simple C++ toy project which by reinforcement learning tries to learn a trajectory of a missile with air resistance.

The way it works is that
1) the missile is launched and its trajectory is computed with numerical ODE solver. We assume some air resistance
2) then we take the final position
3) the final position can be written as a deterministic function of an initial position and speed, say

final_position = f(start_position, start_speed)

since f is smooth, it can be approximated with Taylor expansion. 
4) The reinforcement learning algorithm learns the coefficients of this Taylor expansion
