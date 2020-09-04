// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <iomanip>

#include "cocoa_mpi.h"
#include "randalgms.h"

namespace distropt
{
	// formatted output: "iter epochs   primal_obj    dual_obj    pd_gap    t_rpcd   t_comm     time" 
	void formatted_output(std::ostream &ofs, const int iters, const int epochs, const double primal_obj, const double dual_obj,
		const double t_rpcd, const double t_comm, const double t_cocoa)
	{
		ofs << std::setw(3) << iters
			<< std::setw(6) << epochs
			<< std::fixed << std::setprecision(12)
			<< std::setw(16) << primal_obj
			<< std::setw(16) << dual_obj
			<< std::scientific << std::setprecision(3)
			<< std::setw(12) << primal_obj - dual_obj
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << t_rpcd
			<< std::setw(9) << t_comm
			<< std::setw(9) << t_cocoa
			<< std::endl;
	}

	int cocoa_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y,
		const ISmoothLoss &f, const double lambda, const IUnitRegularizer &g, Vector &wio, const cocoa_params &params)
	{
		int m, mpi_rank;	// number of machines and rank 0, ..., m-1
		MPI_Comm_size(comm_world, &m);
		MPI_Comm_rank(comm_world, &mpi_rank);

		// N is totoal number of examples on all machines, D is the feature dimension
		size_t Ni = X.nrows();
		size_t D = X.ncols();

		// construct CoCoA iterats and auxiliary variables
		Vector v(D), w(D), dv(D);
		Vector a(Ni), a1(Ni), h(Ni), Xw(Ni);
		g.conj_grad(v, w);	// make initialization consistent

		//---------------------------------------------------------------------
		// declare variables for tracking progress and time spent in different activities
		double t_cocoa_start = MPI_Wtime();
		double t_rpcd_start, t_comm_start;
		int vlen = params.max_itrs + 1;
		std::vector<int>    n_epochs(vlen, 0);
		std::vector<double> primal_obj(vlen, 0), dual_obj(vlen, 0);
		std::vector<double> t_rpcd(vlen, 0), t_comm(vlen, 0), t_cocoa(vlen, 0);


		// compute initial primal and dual objective values
		double sum_loss = f.sum_loss(Xw, y);
		double dual_val = f.sum_conjugate(a, y);
		double pd_loss[2] = { sum_loss, dual_val };

		t_comm_start = MPI_Wtime();
		MPI_Allreduce(MPI_IN_PLACE, pd_loss, 2, MPI_DOUBLE, MPI_SUM, comm_world);
		t_comm[0] = MPI_Wtime() - t_comm_start;

		primal_obj[0] = pd_loss[0] / N + lambda*g(w);
		dual_obj[0] = -pd_loss[1] / N - lambda*g.conjugate(v);

		t_cocoa[0] = MPI_Wtime() - t_cocoa_start;

		bool display = params.display && mpi_rank == 0;
		if (display) {
			std::cout << std::endl << "iter epochs   primal_obj       dual_obj       pd_gap    t_rpcd   t_comm     time" << std::endl;
			formatted_output(std::cout, 0, n_epochs[0], primal_obj[0], dual_obj[0], t_rpcd[0], t_comm[0], t_cocoa[0]);
		}

		double nu;
		if (params.avg_or_sum == 'a') { nu = 1.0 / m; }
		else if (params.avg_or_sum == 's') { nu = 1.0; }
		else { nu = 1.0; }		// default using sum

		double sigma = nu*m;
		double lambda_N = lambda*N;

		// CoCoA main loop
		int n_iters = params.max_itrs;
		for (int k = 1; k <= params.max_itrs; k++)
		{
			t_rpcd_start = MPI_Wtime();
			randalgms::rpcd_cocoa(X, y, f, lambda_N, sigma, params.local_itrs, Xw, a, a1, 'p', display);
			t_rpcd[k] = MPI_Wtime() - t_rpcd_start;
			n_epochs[k] = n_epochs[k - 1] + params.local_itrs;

			Vector::elem_subtract(a1, a, h);		//  h := a1 - a
			Vector::axpy(nu, h, a);					//  a := a + nu * h 
			X.aATxby(1.0 / lambda_N, h, 0, dv);		// dv := (1/lambda_N) * X' * h

			// Allreduce to compute sum_i dv[i] across m machines
			t_comm_start = MPI_Wtime();
			int dv_len = int(dv.length());
			MPI_Allreduce(MPI_IN_PLACE, dv.data(), dv_len, MPI_VECTOR_TYPE, MPI_SUM, comm_world);
			t_comm[k] += MPI_Wtime() - t_comm_start;

			Vector::axpy(-nu, dv, v);					// !!! need to use -v for w = grad g*(-v)
			g.conj_grad(v, w);
			X.aAxby(1.0, w, 0, Xw);

			sum_loss = f.sum_loss(Xw, y);
			dual_val = f.sum_conjugate(a, y);
			pd_loss[0] = sum_loss;
			pd_loss[1] = dual_val;
			t_comm_start = MPI_Wtime();
			MPI_Allreduce(MPI_IN_PLACE, pd_loss, 2, MPI_DOUBLE, MPI_SUM, comm_world);
			t_comm[k] += MPI_Wtime() - t_comm_start;
			primal_obj[k] = pd_loss[0] / N + lambda*g(w);
			dual_obj[k] = -pd_loss[1] / N - lambda*g.conjugate(v);

			t_cocoa[k] = MPI_Wtime() - t_cocoa_start;

			if (display)
			{
				formatted_output(std::cout, k, n_epochs[k], primal_obj[k], dual_obj[k], t_rpcd[k], t_comm[k], t_cocoa[k]);
			}

			if (primal_obj[k] - dual_obj[k] <= params.eps_gap)
			{
				n_iters = k;
				break;
			}
		}

		//---------------------------------------------------------------------
		// write progress record in a file
		if (mpi_rank == 0) {
			std::ofstream ofs(params.filename, std::ofstream::out);
			ofs << "iter epochs   primal_obj       dual_obj       pd_gap    t_rpcd   t_comm     time" << std::endl;
			for (int k = 0; k <= n_iters; k++) {
				formatted_output(ofs, k, n_epochs[k], primal_obj[k], dual_obj[k], t_rpcd[k], t_comm[k], t_cocoa[k]);
			}
			ofs.close();
		}

		// return final iterate and number of iterations
		wio.copy(w);
		return n_iters;
	}
}