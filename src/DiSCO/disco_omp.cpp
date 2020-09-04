// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <omp.h>

#include "timer.h"
#include "utils.h"
#include "randalgms.h"
#include "regu_loss.h"
#include "lbfgs_omp.h"
#include "disco_omp.h"

namespace distropt
{
	// formatted output: "iter  pcg_itrs   primal_obj    newton_dec    time" 
	void formatted_output(std::ostream &ofs, const int iters, const double mu, const int epochs, const double stepsize,
		const double primal_obj, const double newton_dec, const double t_newton)
	{
		ofs << std::setw(3) << iters
			<< std::scientific << std::setprecision(1)
			<< std::setw(9) << mu
			<< std::setw(6) << epochs
			<< std::scientific << std::setprecision(1)
			<< std::setw(9) << stepsize
			<< std::fixed << std::setprecision(12)
			<< std::setw(16) << primal_obj
			<< std::scientific << std::setprecision(3)
			<< std::setw(12) << newton_dec
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << t_newton
			<< std::endl;
	}

	int disco_omp(const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, const double lambda,
		const SquaredL2Norm &g, Vector &wio, const int m, const disco_params &params)
	{
		// N is totoal number of examples on all machines, D is the feature dimension
		size_t N = X.nrows();
		size_t D = X.ncols();

		if (y.length() != N || wio.length() != D) {
			throw std::runtime_error("DiSCO: Input/output matrix and vector dimensions do not match.");
		}

		// split X into m submatrices consisting subsets of randomly permuted rows
		std::vector<SubSparseMatrixCSR*> Xi(m);
		std::vector<SubVector*> yi(m);
		Vector w(D), grad(D), Xw(N), drv(N), secdrv(N), w_try(D);
		std::vector<Vector*> wi(m);								// m Vectors of dimension D
		std::vector<SubVector*> Xiw(m), drvi(m), secdrvi(m);	// m SubVectors, each of length Ni

		// use the same random seed by specifying seed=1, otherwise random_device() is used
		std::vector<size_t> rowperm = random_permutation(N, 1);
		//std::vector<size_t> rowperm = random_permutation(N);
		Vector y_perm(y, rowperm);		// permute the y vector to match rows of Xi, used by yi[i]
		size_t row_stride = N / m;
		size_t row_remain = N % m;
		size_t row_start = 0;
		for (int i = 0; i < m; i++) {
			size_t n_rows = i < row_remain ? row_stride + 1 : row_stride;
			std::vector<size_t> subrows(rowperm.begin() + row_start, rowperm.begin() + (row_start + n_rows));
			Xi[i] = new SubSparseMatrixCSR(X, subrows);
			yi[i] = new SubVector(y_perm, row_start, n_rows);

			wi[i] = new Vector(D);
			Xiw[i] = new SubVector(Xw, row_start, n_rows);
			drvi[i] = new SubVector(drv, row_start, n_rows);
			secdrvi[i] = new SubVector(secdrv, row_start, n_rows);
			row_start += n_rows;
		}

		//---------------------------------------------------------------------
		// declare variables for tracking progress and time spent in different activities
		HighResTimer timer;
		std::vector<int>    n_epochs;
		std::vector<double> primal_obj, newton_dec, t_newton;

		// compute initial primal and dual objective values
		if (params.avg_init) {
			w.scale(0);
			for (int i = 1; i < m; i++) {
				wi[i]->copy(wio);
				randalgms::sdca(*Xi[i], *yi[i], f, params.lambda_0, g, params.ini_itrs, params.eps_init, *wi[i], *Xiw[i], 'p', 'd', false);
				w.axpy(1.0 / m, *wi[i]);
			}
		}
		else {
			w.copy(wio);
		}

		// be careful that X and y is not shuffled, but the blocks are defined over shuffled versions!
		X.aAxby(1.0, w, 0, Xw);
		double fw = f.sum_loss(Xw, y) / N + lambda*g(w);
		primal_obj.push_back(fw);
		newton_dec.push_back(0);	// need to compute initial Newton decrement?
		t_newton.push_back(timer.seconds_from_start());
		n_epochs.push_back(0);

		if (params.display) {
			std::cout << std::endl << "iter    mu     pcg stepsize   primal_obj    newton_dec     time" << std::endl;
			formatted_output(std::cout, 0, 0, n_epochs[0], 1.0, primal_obj[0], newton_dec[0], t_newton[0]);
		}

		// construct OffsetQuadratic as unit regularizer for PCG
		SquareLoss q_loss;
		OffsetQuadratic q_regu(D);

		// DiSCO main loop
		Vector r(D), s(D), u(D), v(D), Hu(D), Hv(D), r_scaled(D), r_1(D);
		size_t len0 = yi[0]->length();
		Vector zeros0(len0), a0(len0), secdrv0(len0), Xiw0(len0);
		double mu = params.pcg_mu;
		int n_iters = 0;
		for (int k = 1; k <= params.max_itrs; k++)
		{
			// compute gradient and second derivatives
			X.aAxby(1.0, w, 0, Xw);
			f.derivative(Xw, y, drv);
			X.aATxby(1.0, drv, 0, grad);
			grad.scale(1.0 / N);
			grad.axpy(lambda, w);		// need to include L2 regularization 
			f.second_derivative(Xw, y, secdrv);
			// need to compute preconditioning second derivatives separately
			Xi[0]->aAxby(1.0, w, 0, Xiw0);
			f.second_derivative(Xiw0, *yi[0], secdrv0);

			// PCG solving for H * v = grad,  should try use warm start by previous iterate.
			// compute Hessian-vector product: borrow memory of Xw
			v.scale(0);		// warm start does not work better!
			X.aAxby(1.0, v, 0, Xw);
			Vector::elem_multiply(secdrv, Xw, Xw);
			X.aATxby(1.0, Xw, 0, Hv);
			Hv.scale(1.0 / N);			// Hv = H * v, where H = (1/N) X' * diag(secdrv) * X + lambda * I
			Hv.axpy(lambda, v);			// Hv = H * v, where H = (1/N) X' * diag(secdrv) * X + lambda * I
			r.copy(grad);
			r.axpy(-1.0, Hv);			// r = grad - H * v

			//double pcg_eps = params.beta * sqrt(lambda) * grad.norm2();
			double pcg_eps = params.beta * grad.norm2();
			double alpha, beta, delta;
			double rds, rds_1;
			int pcg_count = 0;
			while (r.norm2() > pcg_eps && pcg_count < params.pcg_itrs) {
				// stochastic preconditioning
				if (params.spc_solver == 's') {
					r_scaled.copy(r);
					r_scaled.scale(1.0 / (lambda + mu));
					q_regu.update_offset(r_scaled);				// q(s) = (1/2)||s - r/(lambda+mu)||^2
					randalgms::sdca(*Xi[0], zeros0, secdrv0, q_loss, lambda + mu, q_regu, params.spc_pass, 1e-6, s, a0, 'p', 'd', params.pcg_display);
				}
				else if(params.spc_solver == 'b') {
					// prepare for calling LBFGS
					auto localfval = [&w, &Xi, &Xiw0, &a0, &secdrv0, &r, lambda, mu](const Vector &x) {
						Xi[0]->aAxby(1.0, x, 0, Xiw0);
						Vector::elem_multiply(Xiw0, Xiw0, a0);
						return 0.5*secdrv0.dot(a0) / a0.length() - x.dot(r) + 0.5*(lambda + mu)*x.dot(x);
					};
					auto localgrad = [&w, &Xi, &Xiw0, &a0, &secdrv0, &r, lambda, mu](const Vector &x, Vector &g) {
						Xi[0]->aAxby(1.0, x, 0, Xiw0);
						Vector::elem_multiply(Xiw0, Xiw0, a0);
						Vector::elem_multiply(Xiw0, secdrv0, Xiw0);
						Xi[0]->aATxby(1.0, Xiw0, 0, g);
						g.scale(1.0 / a0.length());
						g.axpy(-1.0, r);
						g.axpy(lambda + mu, x);
						return 0.5*secdrv0.dot(a0) / a0.length() - x.dot(r) + 0.5*(lambda + mu)*x.dot(x);
					};

					lbfgs_params local_params;
					local_params.eps_grad = params.spc_eps;
					local_params.max_itrs = params.spc_pass;
					local_params.display = false;
					lbfgs_omp(localfval, localgrad, r, s, local_params);
				}
				else {
					// without preconditioning
					s.copy(r);
				}
				
				rds = r.dot(s);

				pcg_count++;
				if (pcg_count == 1) {
					u.copy(s);
				}
				else {
					if (params.pcg_polak) {
						beta = (rds - r_1.dot(s)) / rds_1;
					}
					else {
						beta = rds / rds_1;
					}
					u.scale(beta);
					u.axpy(1.0, s);
				}

				// compute Hessian-vector product
				X.aAxby(1.0, u, 0, Xw);		// borrow memory of Xw
				Vector::elem_multiply(secdrv, Xw, Xw);
				X.aATxby(1.0, Xw, 0, Hu);
				Hu.scale(1.0 / N);
				Hu.axpy(lambda, u);			// need to add regularization: H = (1/N) X' * diag * X + lambda * I

				alpha = rds / u.dot(Hu);
				v.axpy(alpha, u);			// v = v + alpha * u
				r.axpy(-alpha, Hu);			// r = r - alpha * Hu

				// update H*v in order to compute Newton decrement
				Hv.axpy(alpha, Hu);			// Hv = Hv + alpha * Hu

				// need extra r_1 vector for Polak-Ribiere formula for beta
				rds_1 = rds;
				r_1.copy(r);
			}

			// compute Newton decrement and Newton update
			delta = sqrt(v.dot(Hv));
			// w = w - (1/(1+delta)) * v
			//w.axpy(-1.0 / (1 + delta), v);


			// add backtracking line search to make sure!
			// backtracking line search (for nonconvex functions, need Wolfe conditions)
			double stepsize = 1.0 / (1 + delta);
			///*
			double sumloss, ftry, gdv = grad.dot(v);
			for (int lscnt = 0; lscnt < params.btls_max; lscnt++)
			{
				w_try.copy(w);
				w_try.axpy(-stepsize, v);
				X.aAxby(1.0, w_try, 0, Xw);
				sumloss = f.sum_loss(Xw, y);
				// compute new objective values
				ftry = sumloss / N + lambda*g(w_try);
				if (ftry < fw - params.btls_dec*stepsize*gdv) {
					break;
				}
				stepsize *= params.btls_rho;
			}
			w.copy(w_try);
			fw = ftry;
			//*/

			// compute new primal and dual objective values
			//X.aAxby(1.0, w, 0, Xw);
			//double sumloss = f.sum_loss(Xw, y);
			// record time and epochs
			//primal_obj.push_back(sumloss / N + lambda*g(w));
			primal_obj.push_back(fw);
			newton_dec.push_back(delta);
			n_epochs.push_back(pcg_count);
			t_newton.push_back(timer.seconds_from_start());

			if (params.display)
			{
				formatted_output(std::cout, k, mu, n_epochs[k], stepsize, primal_obj[k], newton_dec[k], t_newton[k]);
			}

			if (delta < (1 - params.beta)*sqrt(params.eps_obj))
			{
				n_iters = k;
				break;
			}

			// adaptive tuning of mu
			if (params.pcg_adpt) {
				double T_mu = sqrt(1 + 2 * mu / lambda)*log(2.0 / params.beta / lambda);
				if (pcg_count > T_mu) {
					mu *= 2;
				}
				else {
					mu /= 2;
				}
			}
		}

		// delete memory allocated explicitly
		for (size_t i = 0; i < m; i++) {
			delete Xi[i];
			delete yi[i];
			delete wi[i];
			delete Xiw[i];
			delete drvi[i];
			delete secdrvi[i];
		}

		// --------------------------------------------------------------------------
		// return final iterate and number of iterations
		wio.copy(w);
		return n_iters;
	}
}