// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <iomanip>

#include "paragd_mpi.h"

namespace distropt
{
	// formatted output: "iter   primal_obj  stepsize    t_comm     time" 
	void formatted_output(std::ostream &ofs, const int iters, const double primal_obj, const double step_size,
		const double t_comm, const double t_paragd)
	{
		ofs << std::setw(6) << iters
			<< std::fixed << std::setprecision(12)
			<< std::setw(16) << primal_obj
			<< std::scientific << std::setprecision(3)
			<< std::setw(12) << step_size
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << t_comm
			<< std::setw(9) << t_paragd
			<< std::endl;
	}

	int paragd_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y,
		const ISmoothLoss &f, const double lambda, const IUnitRegularizer &g, Vector &wio, const paragd_params &params)
	{
		int m, mpi_rank;	// number of machines and rank 0, ..., m-1
		MPI_Comm_size(comm_world, &m);
		MPI_Comm_rank(comm_world, &mpi_rank);

		// N is totoal number of examples on all machines, D is the feature dimension
		size_t Ni = X.nrows();
		size_t D = X.ncols();

		// construct gradient descent iterats and auxiliary variables
		Vector w(D), w1(D), dw(D), grad(D);
		Vector Xw(Ni), deriv(Ni), L(Ni);
		X.row_sqrd2norms(L);
		double sum_L = L.sum();
		MPI_Allreduce(MPI_IN_PLACE, &sum_L, 1, MPI_DOUBLE, MPI_SUM, comm_world);
		double avg_L = sum_L / N;

		//---------------------------------------------------------------------
		// declare variables for tracking progress and time spent in different activities
		double t_paragd_start = MPI_Wtime();
		double t_comm_start;
		int vlen = params.max_itrs + 1;
		std::vector<double> primal_obj(vlen, 0), step_sizes(vlen, 0);
		std::vector<double> t_comm(vlen, 0), t_paragd(vlen, 0);

		bool display = params.display && mpi_rank == 0;
		if (display) {
			std::cout << "  iter    primal_obj      stepsize   t_comm     time" << std::endl;
		}

		// compute the initial step size from average squared row norms
		double t = 1.0 / avg_L / f.smoothness();	// initial step_size
		double sum_loss, avg_loss;

		// Parallel GD main loop
		w.copy(wio);
		int n_iters = params.max_itrs;
		double previous_obj = 1.0e+100;
		for (int k = 0; k <= params.max_itrs; k++)
		{
			// compute loss function and gradient using Allreduce
			X.aAxby(1.0, w, 0, Xw);
			sum_loss = f.sum_loss(Xw, y);
			f.derivative(Xw, y, deriv);
			X.aATxby(1.0, deriv, 0, grad);

			t_comm_start = MPI_Wtime();
			int grad_len = int(grad.length());
			MPI_Allreduce(MPI_IN_PLACE, &sum_loss, 1, MPI_DOUBLE, MPI_SUM, comm_world);
			MPI_Allreduce(MPI_IN_PLACE, grad.data(), grad_len, MPI_VECTOR_TYPE, MPI_SUM, comm_world);
			t_comm[k] += MPI_Wtime() - t_comm_start;

			grad.scale(1.0 / N);
			avg_loss = sum_loss / N;
			primal_obj[k] = avg_loss + lambda*g(w);

			if (params.line_search) 
			{
				t *= params.ls_incr;
				while (true)	
				{
					w1.copy(w);
					Vector::axpy(-t, grad, w1);	
					g.prox(t * lambda, w1, w1);		// w1 = prox_(t*lambda)g (w - t * grad)
					Vector::elem_subtract(w1, w, dw);		// dw = w1 - w
					double quad_reduction = grad.dot(dw) + dw.dot(dw) / (2 * t);

					// compute average loss and reduction (for robust of synced break)
					X.aAxby(1.0, w1, 0, Xw);
					sum_loss = f.sum_loss(Xw, y);
					double loss_reduction[2] = { sum_loss, quad_reduction };
					t_comm_start = MPI_Wtime();
					MPI_Allreduce(MPI_IN_PLACE, loss_reduction, 2, MPI_DOUBLE, MPI_SUM, comm_world);
					t_comm[k] += MPI_Wtime() - t_comm_start;

					double avg_loss_w1 = loss_reduction[0] / N;
					quad_reduction = loss_reduction[1] / m;
					if (avg_loss_w1 < avg_loss + quad_reduction)
						break;
					else
						t /= params.ls_decr;
				}
				w.copy(w1);
			}
			else {
				Vector::axpy(-t, grad, w);	
				g.prox(t * lambda, w, w);		// w := prox_(t*lambda)g (w - t * grad)
			}

			step_sizes[k] = t;
			t_paragd[k] = MPI_Wtime() - t_paragd_start;

			if (display) {
				formatted_output(std::cout, k, primal_obj[k], step_sizes[k], t_comm[k], t_paragd[k]);
			}

			// check stopping criteria, reduction of objective value
			if (previous_obj - primal_obj[k] < params.eps_obj )
			{
				n_iters = k;
				break;
			}
			previous_obj = primal_obj[k];
		}

		//---------------------------------------------------------------------
		// write progress record in a file
		if (mpi_rank == 0) {
			std::ofstream ofs(params.filename, std::ofstream::out);
			ofs << "  iter    primal_obj      stepsize   t_comm     time" << std::endl;
			for (int k = 0; k <= n_iters; k++) {
				formatted_output(ofs, k, primal_obj[k], step_sizes[k], t_comm[k], t_paragd[k]);
			}
			ofs.close();
		}

		// return final iterate and number of iterations
		wio.copy(w);
		return n_iters;
	}
}