// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include "timer.h"
#include "utils.h"
#include "randalgms.h"
#include "dane_omp.h"
#include "lbfgs_omp.h"

namespace distropt
{
	// formatted output: "iter  pcg_itrs   primal_obj    newton_dec    time" 
	void formatted_output(std::ostream &ofs, const int iters, const int epochs, const double primal_obj, const double t_dane)
	{
		ofs << std::setw(3) << iters
			<< std::setw(6) << epochs
			<< std::fixed << std::setprecision(12)
			<< std::setw(16) << primal_obj
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << t_dane
			<< std::endl;
	}

	int dane_omp(const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, const double lambda,
		const SquaredL2Norm &g, Vector &wio, const int m, const dane_params &params)
	{
		// N is totoal number of examples on all machines, D is the feature dimension
		size_t N = X.nrows();
		size_t D = X.ncols();

		if (y.length() != N || wio.length() != D) {
			throw std::runtime_error("DANE: Input/output matrix and vector dimensions do not match.");
		}

		// split X into m submatrices consisting subsets of randomly permuted rows
		std::vector<SubSparseMatrixCSR*> Xi(m);
		std::vector<SubVector*> yi(m);
		Vector w(D), grad(D), Xw(N), drv(N);
		std::vector<Vector*> wi(m), gi(m);				// m Vectors of dimension D
		std::vector<SubVector*> Xiw(m), drvi(m);

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
			gi[i] = new Vector(D);
			Xiw[i] = new SubVector(Xw, row_start, n_rows);
			drvi[i] = new SubVector(drv, row_start, n_rows);

			row_start += n_rows;
		}

		//---------------------------------------------------------------------
		// declare variables for tracking progress and time spent in different activities
		HighResTimer timer;
		std::vector<int>    n_epochs;
		std::vector<double> primal_obj, t_dane;

		// compute initial primal and dual objective values
		int sdca_passes = 0;
		if (params.avg_init) {
			w.scale(0);
			for (int i = 1; i < m; i++) {
				wi[i]->copy(wio);
				sdca_passes = randalgms::sdca(*Xi[i], *yi[i], f, lambda, g, params.ini_itrs, params.eps_init, *wi[i], *Xiw[i], 'p', 'd', false);
				w.axpy(1.0 / m, *wi[i]);
			}
		}
		else {
			w.copy(wio);
		}

		// be careful that X and y is not shuffled, but the blocks are defined over shuffled versions!
		X.aAxby(1.0, w, 0, Xw);
		double sum_loss = f.sum_loss(Xw, y);
		primal_obj.push_back(sum_loss / N + lambda*g(w));
		t_dane.push_back(timer.seconds_from_start());
		n_epochs.push_back(sdca_passes);

		if (params.display) {
			std::cout << std::endl << "iter   passes   primal_obj     time" << std::endl;
			formatted_output(std::cout, 0, n_epochs[0], primal_obj[0], t_dane[0]);
		}

		// construct OffsetQuadratic as unit regularizer for PCG
		OffsetQuadratic q_regu(D);

		// DANE main loop
		double mu = params.mu_dane;
		int n_itrs = 0;
		for (n_itrs = 1; n_itrs <= params.max_itrs; n_itrs++)
		{
			// compute global gradient
			X.aAxby(1.0, w, 0, Xw);
			f.derivative(Xw, y, drv);
			X.aATxby(1.0, drv, 0, grad);
			grad.scale(1.0 / N);
			// need to include L2 regularization, but the same appears in both grad and gi[i]
			grad.axpy(lambda, w);

			for (int i = 0; i < m; i++) {
				size_t Ni = yi[i]->length();
				double ci = double(m*Ni) / N;
				Xi[i]->aAxby(1.0, w, 0, *Xiw[i]);
				f.derivative(*Xiw[i], *yi[i], *drvi[i]);
				Xi[i]->aATxby(1.0, *drvi[i], 0, *gi[i]);
				//gi[i]->scale(1.0 / Ni);
				gi[i]->scale(double(m) / N);
				// need to include L2 regularization, but the same appears in both grad and gi[i] 
				gi[i]->axpy(ci*lambda, w);
				//gi[i]->axpy(lambda, w);

				// solve the DANE local problem
				gi[i]->axpy(-1.0, grad);

				if (params.local_solver == 's') {
					gi[i]->axpy(mu, w);
					gi[i]->scale(1.0 / (ci*lambda + mu));
					//gi[i]->scale(1.0 / (lambda + mu));

					q_regu.update_offset(*gi[i]);
					double local_lambda = lambda + mu / ci;
					//double local_lambda = (lambda + mu) / ci;

					sdca_passes = randalgms::sdca(*Xi[i], *yi[i], f, local_lambda, q_regu, params.sdca_itrs, params.sdca_epsilon,
						*wi[i], *Xiw[i], 'p', params.sdca_update, params.sdca_display);
				}
				else
				{
					// prepare for calling LBFGS
					RegularizedLoss localoss(*Xi[i], *yi[i], f.symbol(), lambda, g.symbol());
					auto localfval = [&localoss, &w, &gi, i, ci, mu](const Vector &x) {
						Vector dw(x);
						dw.axpy(-1.0, w);
						return ci*localoss.regu_loss(x) - gi[i]->dot(x) + (mu / 2)*dw.dot(dw);
					};
					auto localgrad = [&localoss, &w, &gi, i, ci, mu](const Vector &x, Vector &g) {
						Vector dw(x);
						dw.axpy(-1.0, w);
						localoss.regu_grad(x, g);
						g.scale(ci);
						g.axpy(-1.0, *gi[i]);
						g.axpy(mu, dw);
						return ci*localoss.regu_loss(x) - gi[i]->dot(x) + (mu / 2)*dw.dot(dw);
					};

					lbfgs_params local_params;
					local_params.eps_grad = 1e-8;
					local_params.display = false;
					sdca_passes = lbfgs_omp(localfval, localgrad, w, *wi[i], local_params);
				}
			}

			w.scale(0);
			for (int i = 0; i < m; i++) {
				w.axpy(1.0 / m, *wi[i]);
			}

			// compute new primal and dual objective values
			X.aAxby(1.0, w, 0, Xw);
			double sum_loss = f.sum_loss(Xw, y);
			// record time and epochs
			primal_obj.push_back(sum_loss / N + lambda*g(w));
			n_epochs.push_back(sdca_passes);
			t_dane.push_back(timer.seconds_from_start());

			if (params.display)
			{
				formatted_output(std::cout, n_itrs, n_epochs[n_itrs], primal_obj[n_itrs], t_dane[n_itrs]);
			}
		}

		// delete memory allocated explicitly
		for (size_t i = 0; i < m; i++) {
			delete Xi[i];
			delete yi[i];
			delete wi[i];
			delete gi[i];
			delete Xiw[i];
			delete drvi[i];
		}

		// --------------------------------------------------------------------------
		// return final iterate and number of iterations
		wio.copy(w);
		return n_itrs;
	}
}