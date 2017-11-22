/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
 num_particles = 101;

  // define normal distributions for sensor noise
  normal_distribution<double> N_x_init(0, std[0]);
  normal_distribution<double> N_y_init(0, std[1]);
  normal_distribution<double> N_theta_init(0, std[2]);

  // init particles
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x;
    p.y = y;
    p.theta = theta;
    p.weight = 1.0;

    // add noise
    p.x += N_x_init(gen);
    p.y += N_y_init(gen);
    p.theta += N_theta_init(gen);

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    // calculate new state
    if (fabs(yaw_rate) < 0.00001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x += N_x(gen);
    particles[i].y += N_y(gen);
    particles[i].theta += N_theta(gen);
  }


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (unsigned int i = 0; i < observations.size(); i++) {
    
    // grab current observation
    LandmarkObs o = observations[i];

    // init minimum distance to maximum possible
    double min_dist = numeric_limits<double>::max();

    // init id of landmark from map placeholder to be associated with the observation
    int map_id = -1;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      // grab current prediction
      LandmarkObs p = predicted[j];
      
      // get distance between current/predicted landmarks
      double cur_dist = dist(o.x, o.y, p.x, p.y);

      // find the predicted landmark nearest the current observed landmark
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        map_id = p.id;
      }
    }

    // set the observation's id to the nearest predicted landmark's id
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {

		// Get the observations coordinates in the map system

		vector<LandmarkObs> trans_observations;
		LandmarkObs obs;
		Particle particle = particles[i];
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs trans_obs;
			obs = observations[j];
			// Aqui puede estar la clave falta el id. No lo creo, pero bueno...
			trans_obs.x = particle.x + (cos(particle.theta) * obs.x) - (sin(particle.theta) * obs.y);
			trans_obs.y = particle.y + (sin(particle.theta) * obs.x) + (cos(particle.theta) * obs.y); 
			trans_observations.push_back(trans_obs);
		}

		// Get the landmarks in the sensor range for the particle

		std::vector<LandmarkObs> filtered;
		for (int j=0; j < map_landmarks.landmark_list.size(); j++) {
			int map_id=map_landmarks.landmark_list[j].id_i;
			float map_x=map_landmarks.landmark_list[j].x_f;
			float map_y=map_landmarks.landmark_list[j].y_f;
			float distance=dist(map_x, map_y, particle.x, particle.y);
			if (distance < sensor_range) {
				LandmarkObs obs_range;
				obs_range.id = map_id;
				obs_range.x = map_x;
				obs_range.y = map_y;
				filtered.push_back(obs_range);
			}
		}

		// Get the associations for the observations

		dataAssociation(filtered, trans_observations);

		// Calculate the new weights for the particle
		particles[i].weight = 1.0;
		// For each observation
		double sig_x=std_landmark[0];
		double sig_y=std_landmark[1];
		for (int j=0; j < trans_observations.size(); j++) {
			// Get the landmark associated
			LandmarkObs observ=trans_observations[j];
			double x = observ.x;
			double y = observ.y;
			double ux, uy;
			for (int k=0; k < filtered.size(); k++) {
				LandmarkObs pos_land=filtered[k];
				if (observ.id == pos_land.id) {
					ux = pos_land.x;
					uy = pos_land.y;
				}
			}

			// Calculate partial weight

			double gauss_norm = (1/(2* M_PI * sig_x * sig_y));
			double exponent1 = ((x - ux) * (x - ux)) / (2 * sig_x * sig_x);
			double exponent2 = ((y - uy) * (y - uy)) / (2 * sig_y * sig_y);
			double exponent = exponent1 + exponent2;
			double total = gauss_norm * exp(-exponent);
			particles[i].weight = particles[i].weight * total;
		}
		weights.push_back(particles[i].weight);
	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	weights.clear();
	vector<Particle> particles_resampled;
	double beta = 0;
	uniform_int_distribution<int> uniintdist(0, num_particles-1);
	std::default_random_engine gen;
	int index = uniintdist(gen);
	double max_weight=*max_element(weights.begin(), weights.end());
	cout << "Max weight: " << max_weight << "\n";
	uniform_real_distribution<double> unirealdist(0.0, 2*max_weight);
	for (int i=0; i < num_particles; i++) {
		beta = beta + unirealdist(gen);
		while (weights[index] < beta) {
			beta = beta - weights[index];
			index = index + 1;
			index = index % num_particles;
		}
		particles_resampled.push_back(particles[index]);
	}
	particles = particles_resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates
	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;
	
 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
