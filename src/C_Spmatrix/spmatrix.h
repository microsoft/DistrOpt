/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */

#ifndef SPMATRIX_H
#define SPMATRIX_H

#include "dnvector.h"

#define SPLIST_INISIZE 100;
#define SPLIST(L) (L && L->count <= L->size && L->count >= 0)
#define SPMAT(A) (A && A->assigned && A->nnz == A->clp[A->ncl])

typedef struct spmatrix_element 
{
	int i;
	int j;
	double x;
} spmat_elem;

typedef struct spmatrix_list
{
	int size;
	int count;
	spmat_elem *elem;
	int sorted;
} spmat_list;

/* functions to operate splist: allocate and free memory, add one element, sorting + summing duplicates*/
spmat_list* spmat_list_alloc(int size);
void		spmat_list_free(spmat_list** pp);
void        spmat_list_reset(spmat_list* pp);
int			spmat_list_add(spmat_list* list, int i, int j, double x);	/* add entry, return 0 if unsuccessful */
int			spmat_list_sort_dupl(spmat_list* list);		/* sort list and sum duplicated elements, return count */

typedef struct spmatrix_csc		/* sparse matrix in compresses sparse column (csc) format */
{
	int nrw;		/* number or rows */
	int ncl;		/* number of columns */
	int nnz;		/* number of non-zero elements */
	int* clp;		/* column pointers (size: ncl + 1) */
	int* rwi;		/* row indices (size: nnz) */
	double* val;	/* non-zero values (size: nnz) */
	int assigned;	/* indicator if elements are assigned values, 0 means allocated but not assigned */
} spmatrix;

/* functions to operate spmatrix: allocate and free memory, copy from three arrays or from a splist */
spmatrix* spmat_alloc(int nrw, int ncl, int nnz);
spmatrix* spmat_alloc_copy(int nrw, int ncl, int nnz, const int* Aclp, const int *Arwi, const double* Aval);
spmatrix* spmat_alloc_splist(int nrw, int ncl, spmat_list* splist);
void      spmat_free(spmatrix** pp);

/* constructing simple sparse matrices */
spmatrix* spmat_zero(int nrw, int ncl);
spmatrix* spmat_diag(const dnvector* v);
spmatrix* spmat_eye(int n);
spmatrix* spmat_row(const dnvector *v); /* constructs a row vector spmatrix from v */
spmatrix* spmat_col(const dnvector *v); /* constructs a column vector spmatrix from v */


/* the following two functions are inefficient, try avoid using them if possible */
int spmat_set_elem(spmatrix* A, int i, int j, double Aij);	/* A(i,j) = Aij, return -1 if (i,j) not found */
int spmat_inc_elem(spmatrix* A, int i, int j, double dlt);	/* A(i,j) +=dlt, return -1 if (i,j) not found */

/* BLAS gemv function: y := alpha*A*x + beta*y and the transpose version: y := alpha*A'*x + beta*y */
void spmat_aAxby(double alpha, const spmatrix* A, const dnvector* x, double beta, dnvector* y);
void spmat_aATxby(double alpha, const spmatrix* A, const dnvector* x, double beta, dnvector* y);

spmatrix *spmat_add(const spmatrix *A, const spmatrix *B); /* C = A + B */
void spmat_add_diagonal(spmatrix **pA, const dnvector *v); /* A = A + diag(v) */

/* functions related to forming augmented system, LDL factorization, and solving linear systems */
spmatrix* spmat_transpose(const spmatrix *A);
spmatrix* spmat_stack_cols(const spmatrix* A, const spmatrix* B);						/* AB = [A B]  */
spmatrix* spmat_stack_rows(const spmatrix* A, const spmatrix* B);						/* AB = [A; B] */
spmatrix* spmat_stack_symm(const spmatrix* A11, const spmatrix* A21, const spmatrix* A22);	/* A = [A11 A21'; A21 A22] */

/* auxiliary functions for debugging */
void spmat_print(const spmatrix* A, int showval);
void spmat_pcprint(const spmatrix *A); /* Print in packed column format. */
void spmat_print2(const spmatrix *A);

spmatrix *spmat_rand(int nrw, int ncl, double density);

void spmat_scale(spmatrix *A, double s); /* A = s A */

#endif