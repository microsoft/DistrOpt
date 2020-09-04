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
#include "disco_omp.h"
#include "lbfgs_omp.h"

namespace distropt
{
	// formatted output: "iter  pcg_itrs   primal_obj    newton_dec    time" 
	void formatted_output_mpi(std::ostream &ofs, const int iters, const int ncomms, const double mu, const int epochs, 
		const double stepsize, const double primal_obj, const double newton_dec, const double t_newton)
	{
		ofs << std::setw(3) << iters
			<< std::setw(6) << ncomms
			<< std::scientific << std::setprecision(1)
			<< std::setw(9) << mu
			<< std::setw(5) << epochs
			<< std::scientific << std::setprecision(1)
			<< std::setw(10) << stepsize
			<< std::fixed << std::setprecision(12)
			<< std::setw(16) << primal_obj
			<< std::scientific << std::setprecision(3)
			<< std::setw(12) << newton_dec
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << t_newton
			<< std::endl;
	}

	int disco_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, 
		const double lambda, const SquaredL2Norm &g, Vector &wio, const disco_params &params)
	{
		int m, mpi_rank;	// number of machines and rank 0, ..., m-1
		MPI_Comm_size(comm_world, &m);
		MPI_Comm_rank(comm_world, &mpi_rank);

		// N is totoal number of examples on all machines, D is the feature dimension
		size_t Ni = X.nrows();
		size_t D = X.ncols();

		if (y.length() != Ni || wio.length() != D) {
			throw std::runtime_error("DiSCO: Input/output matrix and vector dimensions do not match.");
		}

		//---------------------------------------------------------------------
		// declare variables for tracking progress and time spent in different activities
		HighResTimer timer;
		std::vector<int>    n_epochs, n_iters, n_commcnt;
		std::vector<double> primal_obj, newton_dec, t_newton, pcg_mu, stepsizes;

		// create local regularized loss model
		RegularizedLoss localloss(X, y, f.symbol(), lambda, g.symbol());
		Vector w(D), w_try(D), grad(D), Xw(Ni), drv(Ni), secdrv(Ni), zeros(Ni), a(Ni);

		w.copy(wio);

		// first compute overall loss
		double sumloss = localloss.sum_loss(w);
		MPI_Allreduce(MPI_IN_PLACE, &sumloss, 1, MPI_DOUBLE, MPI_SUM, comm_world);
		double fw = sumloss / N + lambda*g(w);

		primal_obj.push_back(fw);
		newton_dec.push_back(0);	// need to compute initial Newton decrement?
		t_newton.push_back(timer.seconds_from_start());
		n_epochs.push_back(0);
		n_iters.push_back(0);
		n_commcnt.push_back(0);
		pcg_mu.push_back(params.pcg_mu);
		stepsizes.push_back(1);

		bool display = params.display && mpi_rank == 0;
		if (display) {
			std::cout << std::endl << "iter  comms    mu    pcg stepsize    primal_obj    newton_dec     time" << std::endl;
			formatted_output_mpi(std::cout, 0, 1, 0, n_epochs[0], 0, primal_obj[0], newton_dec[0], t_newton[0]);
		}

		// construct OffsetQuadratic as unit regularizer for PCG
		SquareLoss q_loss;
		OffsetQuadratic q_regu(D);

		// DiSCO main loop
		Vector r(D), s(D), u(D), v(D), Hu(D), Hv(D), r_scaled(D), r_1(D);

		double mu = params.pcg_mu;
		int n_newton = 0;
		int n_comms = 0;
		for (int k = 1; k <= params.max_itrs; k++)
		{
			// compute gradient and second derivatives
			X.aAxby(1.0, w, 0, Xw);
			f.derivative(Xw, y, drv);
			X.aATxby(1.0, drv, 0, grad);
			MPI_Allreduce(MPI_IN_PLACE, grad.data(), grad.length(), MPI_VECTOR_TYPE, MPI_SUM, comm_world);
			grad.scale(1.0 / N);
			// need to include L2 regularization, but the same appears in both grad and gi[i]
			grad.axpy(lambda, w);
			n_comms++;

			// PCG solving for H * v = grad,  should try use warm start by previous iterate.
			// compute Hessian-vector product: Hessian at w, so need to use the same Xw computed above
			f.second_derivative(Xw, y, secdrv);
			v.scale(0);					// warm start does not work better!
			X.aAxby(1.0, v, 0, Xw);		// borrow memory of Xw is okay because secdrv is what we need later
			Vector::elem_multiply(secdrv, Xw, Xw);
			X.aATxby(1.0, Xw, 0, Hv);
			MPI_Allreduce(MPI_IN_PLACE, Hv.data(), Hv.length(), MPI_VECTOR_TYPE, MPI_SUM, comm_world);
			Hv.scale(1.0 / N);			// Hv = H * v, where H = (1/N) X' * diag(secdrv) * X + lambda * I
			Hv.axpy(lambda, v);			// Hv = H * v, where H = (1/N) X' * diag(secdrv) * X + lambda * I
			r.copy(grad);
			r.axpy(-1.0, Hv);			// r = grad - H * v

			//double pcg_eps = params.beta * sqrt(lambda) * grad.norm2();
			double pcg_eps = params.beta * grad.norm2();
			double alpha, beta, delta;
			double stepsize = 1;
			double rds, rds_1;
			int pcg_count = 0;
			while (r.norm2() > pcg_eps && pcg_count < params.pcg_itrs) {
				// stochastic preconditioning
				if (mpi_rank == 0) {
					r_scaled.copy(r);
					r_scaled.scale(1.0 / (lambda + mu));
					q_regu.update_offset(r_scaled);				// q(s) = (1/2)||s - r/(lambda+mu)||^2
					randalgms::sdca(X, zeros, secdrv, q_loss, lambda + mu, q_regu, params.spc_pass, 1e-6, s, a, 'p', 'd', params.pcg_display);
				}
				// only process 0 perform preconditioning, then broadcast to everyone
				MPI_Bcast(s.data(), s.length(), MPI_VECTOR_TYPE, 0, comm_world);

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
				MPI_Allreduce(MPI_IN_PLACE, Hu.data(), Hu.length(), MPI_VECTOR_TYPE, MPI_SUM, comm_world);
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

				n_comms++;

				// record inner iteration objective values
				if (params.pcg_record) {
					delta = sqrt(v.dot(Hv));
					// w = w - (1/(1+delta)) * v
					w_try.copy(w);
					w_try.axpy(-1.0 / (1 + delta), v);

					// compute new primal and dual objective values
					sumloss = localloss.sum_loss(w_try);
					MPI_Allreduce(MPI_IN_PLACE, &sumloss, 1, MPI_DOUBLE, MPI_SUM, comm_world);

					// record time and epochs
					primal_obj.push_back(sumloss / N + lambda*g(w_try));
					newton_dec.push_back(delta);
					n_epochs.push_back(pcg_count);
					n_iters.push_back(k);
					n_commcnt.push_back(n_comms);
					t_newton.push_back(timer.seconds_from_start());
					pcg_mu.push_back(mu);
					stepsizes.push_back(stepsize);
				}
			}

			// compute Newton decrement and Newton update
			delta = sqrt(v.dot(Hv));

			stepsize = 1.0 / (1 + delta);

			if (!params.linesearch) {
				// w = w - (1/(1+delta)) * v
				w.axpy(-1.0 / (1 + delta), v);

				// compute new primal and dual objective values
				sumloss = localloss.sum_loss(w);
				MPI_Allreduce(MPI_IN_PLACE, &sumloss, 1, MPI_DOUBLE, MPI_SUM, comm_world);
				fw = sumloss / N + lambda*g(w);
			}
			else {
				// add backtracking line search to make sure!
				// backtracking line search (for nonconvex functions, need Wolfe conditions)
				double ftry, gdv = grad.dot(v);
				for (int lscnt = 0; lscnt < params.btls_max; lscnt++)
				{
					w_try.copy(w);
					w_try.axpy(-stepsize, v);
					// compute new primal and dual objective values
					sumloss = localloss.sum_loss(w_try);
					MPI_Allreduce(MPI_IN_PLACE, &sumloss, 1, MPI_DOUBLE, MPI_SUM, comm_world);
					ftry = sumloss / N + lambda*g(w_try);
					if (ftry < fw - params.btls_dec*stepsize*gdv) {
						break;
					}
					stepsize *= params.btls_rho;
				}
				w.copy(w_try);
				fw = ftry;
			}

			if (!params.pcg_record) {
				// record time and epochs
				primal_obj.push_back(fw);
				newton_dec.push_back(delta);
				n_epochs.push_back(pcg_count);
				n_iters.push_back(k);
				n_commcnt.push_back(n_comms);
				t_newton.push_back(timer.seconds_from_start());
				pcg_mu.push_back(mu);
				stepsizes.push_back(stepsize);
			}

			if (display)
			{
				int last = primal_obj.size() - 1;
				formatted_output_mpi(std::cout, k, n_comms, mu, n_epochs[last], stepsize, primal_obj[last], newton_dec[last], t_newton[last]);
			}

			n_newton = k;
			if (delta < (1 - params.beta)*sqrt(params.eps_obj))
			{
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

		//---------------------------------------------------------------------
		// write progress record in a file
		if (mpi_rank == 0) {
			std::ofstream ofs(params.filename, std::ofstream::out);
			ofs << "iter  comms    mu    pcg stepsize    primal_obj    newton_dec     time" << std::endl;
			for (int k = 0; k < primal_obj.size(); k++) {
				formatted_output_mpi(ofs, n_iters[k], n_commcnt[k], pcg_mu[k], n_epochs[k], stepsizes[k], primal_obj[k], newton_dec[k], t_newton[k]);
			}
			ofs.close();
		}

		// --------------------------------------------------------------------------
		// return final iterate and number of iterations
		wio.copy(w);
		return n_newton;
	}
}