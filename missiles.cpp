#include <iostream>
#include <random>
#include <cstdlib>
#include <utility>
#include <cmath>
#include <fstream>
#include <array>
#include "my_array.h"
using namespace std;

// TODO - reward computation

double GRAVITY = 9.81;


double my_unif(){
	double number = rand();
	return number / RAND_MAX;
}

double my_gaussian(){
	double first, second, radius;
	first = my_unif();
	second = my_unif();
	radius = sqrt(-2.0 * log(first));

	return radius * sin(2.0 * 3.14 * second);
}

// we are in a continuous world, exact zero will not happen anyway
template <typename T>
T my_sign(T x){
	if (x>=0)
		return 1;
	return -1;
};

// necessary? is it possible to use templates or something like it ?
double my_div(int k, double m){
	return (double) k / m;
};

double my_div(double m, int k){
	return m / (double) k;
};

double my_div(int m, int k){
	return (double) m / (double) k;
}

template <typename T>
T dot_product(std::vector<T> x, std::vector<T> y){
	assert(x.size() == y.size());
	int N = x.size();
	T s = 0;
	for (int i = 0; i < N; i++){
		s += x[i] * y[i];
	};
	return s;
}

std::vector<double> two_gaussians(){
	double first, second, radius;
	first = my_unif();
	second = my_unif();

	radius = sqrt(-2.0 * log(first));

	std::vector<double> gaus_array;
	gaus_array.push_back(radius * sin(2.0 * 3.14 * second));
	gaus_array.push_back(radius * cos(2.0 * 3.14 * second));
	return gaus_array;
};

std::vector<double> compute_column_mean_with_idx(std::vector<std::vector<int> > &array){
	cout << "in computing means " << endl;
	int n = array[0].size();
	int length = array.size();
	std::vector<double> means;

	for (int i = 0; i < n; i++)
		means.push_back(0);

	for (int j = 0; j < length; j++){
		for (int i = 0; i < n; i++){
			means[i] = (1 - 1/(double)(j+1))*means[i] + (double) array[j][i] / (j+1);
		};
	};

	return means;
};

// line is there because Array is going to be 3d
std::vector<double> compute_local_peaks(Array &im, int line, int radius){
	int mrad = (int)(radius/2);
	assert(im.n_dim == 3);
	std::vector<std::vector<int> > list_of_peaks;
	std::vector<double> final_peaks;

	for (int i = 0; i < im.dimensions[1]; i++){
		for (int j = 0; j < im.dimensions[2]; j++){

			double local_max, loclocval;
			bool flat_patch(1);
			local_max = im.get_value(line, i, j); // essentially we start from the current value, if we find something that is bigger than it's not a peak
			for (int k = 0; k < radius; k++){
				for (int m = 0; m < radius; m++){

					if (i - mrad + k >= 0 and i - mrad + k < im.dimensions[1] and j - mrad + m >= 0 and j - mrad + m < im.dimensions[2]){

						loclocval = im.soft_get_value(line, i - mrad + k, j - mrad + m); 
						local_max = max(local_max, loclocval);
						if (im.get_value(line, i, j) != loclocval)
							flat_patch = 0;
					};
				};
			};

			if (!flat_patch and (local_max <= im.get_value(line, i, j) )){
				std::vector<int> lp;
				lp.push_back(i);
				lp.push_back(j);
				list_of_peaks.push_back(lp);
				cout << "add new peak " << endl;
			}
		};
	};

	cout << "peaks computed, nr " << list_of_peaks.size() <<  endl;
	if (list_of_peaks.size() == 1){
		final_peaks.push_back( (double) list_of_peaks[0][0] );
		final_peaks.push_back( (double) list_of_peaks[0][1] );
		return final_peaks;
	} else
		return compute_column_mean_with_idx(list_of_peaks);
};

class Missile{
public:
	double pos_x, pos_y, vel_x, vel_y; // design choice - do I replace it with Arrays?

	Missile(double px, double py, double vx, double vy);

	void fix_speed(std::vector<double> v);
	void reset();

	// euler schemes update steps - we return 1 if we have not hit the ground yet
	bool update_explicit(double timestep, bool random_perturbation, double air_resistance, double variance); 
	bool update_implicit(double timestep, bool random_perturbation, double air_resistance);

	// integration of the whole trajectory
	void integrate_explicit(double timestep, bool random_perturbation, double air_resistance, Array& arr, double variance, int N);
	void integrate_explicit_no_array(double timestep, bool random_perturbation, double air_resistance, double variance, int N);
};

Missile::Missile(double px, double py, double vx, double vy){
	pos_x = px; // positions.input_value(0) = px;
	pos_y = py; // positions.input_value(1) = py
	vel_x = vx; // speeds.input_value(0) = vx;
	vel_y = vy; // speeds.input_value(1) = vy;
};

void Missile::fix_speed(std::vector<double> v){
	vel_x = v[0];
	vel_y = v[1];
};

void Missile::reset(){
	pos_x = 0;
	pos_y = 0;
	vel_x = 0;
	vel_y = 0;
}

bool Missile::update_explicit(double timestep, bool random_perturbation, double air_resistance, double variance = 1){
	double fx, fy;
	fy = -GRAVITY - my_sign(vel_y) * air_resistance * pow(vel_y, 2);
	fx = - my_sign(vel_x) * air_resistance * pow(vel_x, 2);

	vel_x = vel_x + fx * timestep;
	vel_y = vel_y + fy * timestep;

	pos_x = pos_x + vel_x * timestep;
	pos_y = pos_y + vel_y * timestep;

	if (random_perturbation){
		std::vector<double> pert = two_gaussians();
		pos_x += pert[0] * timestep * variance;
		pos_y += pert[1] * timestep * variance;
	};

	return (pos_y >= 0);
};

void Missile::integrate_explicit(double timestep, bool random_perturbation, double air_resistance, Array& arr, double variance = 1, int N = 1000){
	bool cont = 1;
	int n = 0;
	arr.insert_value(n, {pos_x, pos_y, vel_x, vel_y});

	while (cont & (n++ < N)){
		cont = update_explicit(timestep, random_perturbation, air_resistance, variance);
		arr.insert_value(n, {pos_x, pos_y, vel_x, vel_y});
	};
};

void Missile::integrate_explicit_no_array(double timestep, bool random_perturbation, double air_resistance, double variance = 1, int N = 1000){
	bool cont = 1;
	int n = 0;

	while (cont & (n++ < N)){
		cont = update_explicit(timestep, random_perturbation, air_resistance, variance);
	};
};

class Player{
public:
	double pos_x;
	double timestep;
	char player_type;

	// method that will sample parameters to put in the missile (that is initial speed)
	std::vector<double> sample_params();

	// method that will update our estimate of parameters parameters. Will always be replaced by an inheriting class (TODO now it's not really proper inheritance)
	void update_params();
	void compute_reward(double vel_x, double vel_y, double target_x);

	void update_params_re(std::vector<double> &, double){};
	void update_params_avg(Array &){};

private:
	double air_resistance_estimate;
};

// TODO CHANGE, it should give speeds
std::vector<double> Player::sample_params(){
	return {20, 20};
};


// PROBABLY ABANDONED, NOT INTERESTING
// we will take the physics of the object and compute the estimate for the missing parameters (air resistance)
// of course there are degrees of how easy it can be - we will assume that we have access to video information 
// therefore we will be able to estimate it quickly
class AveragePlayer : public Player{
public:
	AveragePlayer(int radius, int k_images, int resolution_multiplier, double timestep);

	std::vector<double> peak_to_position(std::vector<double> lp);
	std::vector<int> position_to_peak(std::vector<double> pos);
	Array simulate_image_path(Array &, int, int);
	void update_params_avg(Array &);
	char player_type;

private:
	int k_images;
	int radius;
	int resolution_multiplier; // this parameter will rule how 
	Array image_sequence;
	double air_resistance_estimate;
};

AveragePlayer::AveragePlayer(int radius, int k_images, int resolution_multiplier, double timestep){
	// I'm using this because otherwise it does not register the change, don't know why
	this->k_images = k_images;
	this->radius = radius;
	this->resolution_multiplier = resolution_multiplier;
	this->timestep = timestep;
	this->player_type = 'A';
};

// maybe this should be improved
Array AveragePlayer::simulate_image_path(Array &path_sequence, int imsizex, int imsizey){
	Array image_sequence(k_images, imsizex, imsizey);
	int l = path_sequence.dimensions[0] - 1;

	std::vector<int> positions;
	for (int i = 0; i < k_images; i++){
		positions = position_to_peak({path_sequence.get_value(i, 0), path_sequence.get_value(i, 1)});
		cout << "position " << i << ": " << path_sequence.get_value(i, 0) << " -> " << positions[0] << ", " << path_sequence.get_value(i, 1) << " -> " << positions[1] << endl;
		image_sequence.insert_value(i, positions[0], positions[1], 1); // this should be a blob, to be adapted from a previous thing
	};
	return image_sequence;
};

// we will simply assume that AvgPlayer launches from zero, it will be easier to adapt the other one I think
void AveragePlayer::update_params_avg(Array &path_sequence){
	int sizex(0), sizey(0);
	for (int i = 0; i < path_sequence.dimensions[0]; i++){
		sizex = std::max<int>(sizex, path_sequence.get_value(i, 0));
		sizey = std::max<int>(sizey, path_sequence.get_value(i, 1));
	};

	sizex += 2;
	sizey += 2;

	cout << "we now simulate images" << endl;
	Array image_sequence = simulate_image_path(path_sequence, resolution_multiplier * sizex, resolution_multiplier * sizey);

	std::vector<double> speeds_x, speeds_y;
	double est_air_rest_x(0), est_air_rest_y(0);

	Array peaks(k_images, 2);

	// code getting first k positions here
	// also actually we would aim at getting something slightly more realistic, like reading out the position from the difference, so this is to be changed
	// you can adapt the existing code from the first project
	cout << "we now compute peaks" << endl;

	for (int i = 0; i < k_images; i++){
		std::vector<double > local_peaks;
		local_peaks = compute_local_peaks(image_sequence, i, radius); // std::vector<double>
		cout << "image nr " << i << ", peak is: " << local_peaks[0] << ", " << local_peaks[1] << endl;
		local_peaks = peak_to_position(local_peaks); // std::vector<double>
		peaks.insert_value(i, local_peaks);
		cout << "image nr " << i << ", discovered position is " << local_peaks[0] << ", " << local_peaks[1] << endl;
	};

	cout << "speeds " << endl;
	// code for estimating speed here
	// I could probably do without this intermediary step and do everything on positions that we know
	for (int i = 0; i < k_images-1; i++){
		speeds_x.push_back((peaks.get_value(i+1, 0) - peaks.get_value(i, 0)) / this->timestep);
		speeds_y.push_back((peaks.get_value(i+1, 1) - peaks.get_value(i, 0)) / this->timestep);
		cout << "speed x,y: " << speeds_x[i] << ", " << speeds_y[i] << endl;
	};

	cout << "inverse physics" << endl;
	// put "inverse physics" here. We will infer the parameters from the first k images.
	for (int j = 0; j < k_images - 2; j++){
		double weight = my_div(1, j+1);
		est_air_rest_x = est_air_rest_x * (1-weight) + weight * ((speeds_x[j+1] - speeds_x[j]) / timestep) / pow(speeds_x[j], 2);
		est_air_rest_y = est_air_rest_y * (1-weight) + weight * ((speeds_y[j+1] - speeds_y[j]) / timestep - GRAVITY) / pow(speeds_y[j], 2); 
	};

	this->air_resistance_estimate = est_air_rest_y / 2 + est_air_rest_x / 2;
};

std::vector<double> AveragePlayer::peak_to_position(std::vector<double> lp){
	return {my_div(lp[0], resolution_multiplier), my_div(lp[1], resolution_multiplier)};
};

std::vector<int> AveragePlayer::position_to_peak(std::vector<double> pos){
	return {(int) (pos[0] * resolution_multiplier), (int) (pos[1] * resolution_multiplier )};
};

// reinforcement learning - for any given policy (i.e. speeds) we should have a model of what award is that going to give
// therefore we sample a policy, we then compare it with our estimate and update via SGD
// for simplicity we try to make it as linear as possible - to this end we will actually try to approximate coefficients in Taylor approximation
class ReinforcedPlayer : public Player{
public:

	ReinforcedPlayer(int, double, double, double);

	std::vector<double> coefficients;
	std::vector<std::vector<int> > binomial_coefs;
	void update_params_re(std::vector<double>&, double);
	std::vector<double> sample_params();
	double predict(std::vector<double>);
	char player_type;

private:
	int taylor_step;
	double learning_rate;
	double sampling_up;
	double sampling_low;
};

int binom(int n,int k, std::vector<std::vector<int> > &binomial_coefs){
	if (k > n)
		return 0;

	if (k == 0 || k ==n)
		return 1;

	if (binomial_coefs[n][k] != -1)
		return binomial_coefs[n][k];

	binomial_coefs[n][k] = binom(n-1,k-1, binomial_coefs) + binom(n-1,k, binomial_coefs);
	return binomial_coefs[n][k];
};

ReinforcedPlayer::ReinforcedPlayer(int taylor_step, double learning_rate, double sampling_low, double sampling_up){
	this->taylor_step = taylor_step;
	this->learning_rate = learning_rate;
	this->player_type = 'R';
	this->sampling_up = sampling_up;
	this->sampling_low = sampling_low;

	int size_of_bc = 0;
	int c = 0;
	for (int i = 0; i < taylor_step+1; i++){
		std::vector<int> v;
		for (int j = 0; j < taylor_step+1; j++){
			v.push_back( -1);
		};
		binomial_coefs.push_back(v);
	};

	binom(taylor_step, taylor_step, this->binomial_coefs);

	int length_of_coef = (taylor_step+3) * (taylor_step) / 2;
	for (int i = 0; i < length_of_coef; i++){
		this->coefficients.push_back(my_gaussian() / 100);
	};

	cout << "Created Player: " << endl;
	cout << "length of coefficients: " << length_of_coef << endl;
	cout << "taylor step is " << this->taylor_step << endl;
};

std::vector<double> ReinforcedPlayer::sample_params(){
	return {sampling_low + my_unif() * (sampling_up - sampling_low), sampling_low + my_unif() * (sampling_up - sampling_low)};
};

double ReinforcedPlayer::predict(std::vector<double> v){

	std::vector<double> approx_vector;
	for (int i = 1; i < this->taylor_step+1;i++){
		for (int j =0; j < i+1; j++){
			approx_vector.push_back( binomial_coefs[i][j] * pow(v[0], j) * pow(v[1], i-j) );
		};
	};

	return dot_product(approx_vector, this->coefficients);
};

void ReinforcedPlayer::update_params_re(std::vector<double> &init_velocity, double real_last_pos){

	double prediction;
	std::vector<double> approx_vector;
	for (int i = 1; i < this->taylor_step+1;i++){
		for (int j =0; j < i+1; j++){
			approx_vector.push_back( binomial_coefs[i][j] * pow(init_velocity[0], j) * pow(init_velocity[1], i-j) );
		};
	};

	prediction = dot_product(approx_vector, this->coefficients);

	for (int i =0; i < approx_vector.size(); i++){
		this->coefficients[i] -= this->learning_rate * (prediction - real_last_pos) * approx_vector[i];
	};
};

double compute_endmean(std::vector<double> v, int m){
	int n = v.size()-1;
	assert(n > m);
	double s(0);
	for (int i = 0; i < m; i++){
		s += v[n-i];
	};
	return s / m;
}

// the easiest would be to write two versions, done is better than perfect
void run_the_game_average(){
	// TODO
};

void run_the_game_reinforce(ReinforcedPlayer &player, int N, int verbosity, double timestep, double real_air_resistance, 
	bool random_perturbation, double variance, int sim_length = 1000){
	std::vector<double> init_velocity;
	std::vector<double> errors;
	Missile mis(0,0,0,0);
	Array res(sim_length, 4);
	double real_outcome, pred_outcome;
	bool modif_real_outcome(0);
	for (int i = 1; i < N; i++){

		init_velocity = player.sample_params();
		// cout << "init velocity: " << init_velocity[0] << " " << init_velocity[1] << endl;
		mis.fix_speed(init_velocity);
		mis.integrate_explicit(timestep, random_perturbation, real_air_resistance, res, variance);
		for (int j = 0; j < res.dimensions[0]-1; j++){
			if (res.get_value(j, 1) > 0 and res.get_value(j+1,1) < 0){
				real_outcome = res.get_value(j, 0);
				modif_real_outcome = 1;
			};
		};

		// res.dump_to_textfile("zero_res_" + std::to_string(i) + ".txt");
		pred_outcome = player.predict(init_velocity);
		player.update_params_re(init_velocity, real_outcome); // instead you can have some global update_params
		errors.push_back( std::abs(real_outcome - pred_outcome) );
		if (i % verbosity == 0){
			cout << "prediction: " << pred_outcome << ", real " << real_outcome << endl;
			cout << "real outcome with no resistance would be " << 2 * init_velocity[0] * init_velocity[1] / GRAVITY << endl;
			cout << "error mean " << compute_endmean(errors, verbosity-2) << endl;
			cout << "init speeds " << init_velocity[0] << " " << init_velocity[1] << endl << endl;
		};

		modif_real_outcome = 0;
		mis.reset();
		res.clear();
	};
};

int main() {	
	srand((unsigned) time(NULL));

	ReinforcedPlayer player(3, 1e-4, 1, 4);

	run_the_game_reinforce(player, 10000, 1000, 0.001, 0.1, 0, 0);

	// Missile mis(0,0,5,5);
	// Array results(50);
	// results = mis.integrate_explicit(0.01, 0, 0.01, 0);
	// results.dump_to_textfile("zero_res.txt");	
};