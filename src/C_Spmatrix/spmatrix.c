/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "spmatrix.h"

/* Need to change all assert() statements into specific error messages and call exit() */
/* Allocates a spmat_list of size. Returns NULL if not enough memory. */
/* If size = -1, allocates a default size. */
spmat_list* spmat_list_alloc(int size)
{
	spmat_list* list = NULL;

	if (size == -1)
		size = SPLIST_INISIZE;

	list = malloc(sizeof(spmat_list));
	if (!list) return NULL;
	
	list->size = size;
	list->elem = malloc((list->size) * sizeof(spmat_elem));
	if (!list->elem) { free(list); return NULL; }

	list->count = 0;
	list->sorted = 1;
	return list;
}

void spmat_list_free(spmat_list** pp)
{
	if (!(*pp)) return;
	if ((*pp)->elem) free((*pp)->elem);
	free(*pp);
	*pp = NULL;
}

void spmat_list_reset(spmat_list* list)
{
	list->count = 0;
}


int spmat_list_add(spmat_list *list, int i, int j, double x)
{
	int cnt;
	spmat_elem *newptr = NULL;
	assert(SPLIST(list) && i >= 0 && j >= 0);

	cnt = list->count;
	if (cnt == list->size) {
		newptr = realloc(list->elem, (2 * list->size) * sizeof(spmat_elem));
		if (!newptr) return 0;
		list->elem = newptr;
		list->size *= 2;
	}
	list->elem[cnt].i = i;
	list->elem[cnt].j = j;
	list->elem[cnt].x = x;
	list->count++;
	list->sorted = 0;
	return 1;
}

/* compare(a,b), where *a and *b are of type spmat_elem, used by splist_sort_dupl() */
int spmat_elem_compare(const void* a, const void* b)
{
	spmat_elem *e1, *e2;

	assert(a && b);
	e1 = (spmat_elem*)a;
	e2 = (spmat_elem*)b;
	
	return e1->j == e2->j ? e1->i - e2->i : e1->j - e2->j;
}

/* This is a very efficient implementation. Quiz: explain how it works */
int spmat_list_sort_dupl(spmat_list* list)	/* sort list and sum up duplicate elements */
{
	int i, j, cnt;
	assert(SPLIST(list));

	if (list->sorted) return list->count;

	qsort(list->elem, list->count, sizeof(spmat_elem), spmat_elem_compare);
	cnt = list->count;
	i = 0;
	for (j = 1; j < cnt; j++){
		if (spmat_elem_compare(list->elem + i, list->elem + j) == 0) {
			list->elem[i].x += list->elem[j].x;
		}
		else {
			i++;
			if (i < j) list->elem[i] = list->elem[j];
		}
	}
	list->count = i + 1;
	list->sorted = 1;
	return list->count;
}

spmatrix* spmat_alloc(int nrw, int ncl, int nnz)
{
	spmatrix *A = NULL;

	assert(nrw >= 0);
	assert(ncl >= 0);
	assert(nnz >= 0);
	assert(nnz <= nrw * ncl);

	A = malloc(sizeof(spmatrix));
	if (!A) return NULL;

	A->nrw = nrw;
	A->ncl = ncl;
	A->nnz = nnz;
	A->clp = malloc((ncl + 1) * sizeof(int));
	A->rwi = malloc(nnz * sizeof(int));
	A->val = malloc(nnz * sizeof(double));
	if (!A->clp || !A->rwi || !A->val) { spmat_free(&A); return NULL; }
	if (nnz == 0) { A->rwi = NULL; A->val = NULL; }
	A->assigned = 0;
	return A;
}

/* This does not copy a zero matrix correctly. */
spmatrix* spmat_alloc_copy(int nrw, int ncl, int nnz, const int* Aclp, const int *Arwi, const double* Aval)
{
	spmatrix *A = NULL;

	assert(nrw >= 0);
	assert(ncl >= 0);
	assert(nnz >= 0);
	assert(nnz <= nrw * ncl);

	assert(Aclp);
	assert(Aclp[0] == 0);
	assert(Aclp[ncl] == nnz);
	if (nnz >= 1) {
		assert(Arwi);
		assert(Aval);
	}

	A = spmat_alloc(nrw, ncl, nnz);
	if (!A) return NULL;
	memcpy(A->clp, Aclp, (ncl + 1) * sizeof(int));
	if (nnz >= 1) {
		memcpy(A->rwi, Arwi, nnz * sizeof(int));
		memcpy(A->val, Aval, nnz * sizeof(double));
	}

	A->assigned = 1;
	return A;
}

void spmat_free(spmatrix** pp)
{
	if (!(*pp)) return;
 	if ((*pp)->clp) free((*pp)->clp);
	if ((*pp)->rwi) free((*pp)->rwi);
	if ((*pp)->val) free((*pp)->val);
	free(*pp);
	*pp = NULL;
}

spmatrix* spmat_zero(int nrw, int ncl)
{
	spmatrix *Z = NULL;

	assert(nrw >= 0);
	assert(ncl >= 0);

	Z = spmat_alloc(nrw, ncl, 0);
	if (!Z) return NULL;

	memset(Z->clp, 0, (ncl + 1) * sizeof(int));
	Z->assigned = 1;
	return Z;
}

spmatrix* spmat_diag(const dnvector* v)
{
	int j, n;
	assert(DNVEC(v));

	n = v->len;
	spmatrix *V = spmat_alloc(n, n, n);
	if (!V) return NULL;

	for (j = 0; j < n; j++){
		V->clp[j] = j;
		V->rwi[j] = j;
	}
	V->clp[n] = n;

	if (n >= 1)
		memcpy(V->val, v->val, n * sizeof(double));

	V->assigned = 1;
	return V;
}

spmatrix* spmat_eye(int n)
{
	spmatrix *I = NULL;
	dnvector *ones = NULL;
	
	ones = dnvec_alloc(n);
	if (!ones) return NULL;
	dnvec_set_const(ones, 1.0);
	I = spmat_diag(ones);
	dnvec_free(&ones);

	return I;
}

spmatrix * spmat_row(const dnvector * v)
{
	spmatrix *row = NULL;
	int i;

	row = spmat_alloc(1, v->len, v->len);
	if (row == NULL) { return NULL; }

	for (i = 0; i < v->len; i++) {
		row->clp[i] = i;
		row->rwi[i] = 0;
		row->val[i] = v->val[i];
	}
	row->clp[v->len] = v->len;
	row->assigned = 1;

	return row;
}

spmatrix * spmat_col(const dnvector * v)
{
	spmatrix *col = NULL;
	int i;

	col = spmat_alloc(v->len, 1, v->len);
	if (col == NULL) { return NULL; }

	col->clp[0] = 0;
	col->clp[1] = v->len;
	for (i = 0; i < v->len; i++) {
		col->rwi[i] = i;
		col->val[i] = v->val[i];
	}
	col->assigned = 1;

	return col;
}

spmatrix* spmat_alloc_splist(int nrw, int ncl, spmat_list* list)
{
	int k, j, nnz, col, colpre, *Aclp, *Arwi;
	double *Aval;
	spmat_elem *elem;
	spmatrix *A;

	assert(SPLIST(list));

	if (!(list->sorted)) spmat_list_sort_dupl(list);
	nnz = list->count;

	A = spmat_alloc(nrw, ncl, nnz);
	if (!A) return NULL;

	Aclp = A->clp; Arwi = A->rwi; Aval = A->val;
	elem = list->elem;
	colpre = -1; col = -1;
	for (k = 0; k < nnz; k++) {
		Arwi[k] = elem[k].i;
		Aval[k] = elem[k].x;
		col = elem[k].j;
		assert(col < ncl && Arwi[k] < nrw);
		/* need to use loop to take care of empty columns */
		if (col > colpre) {
			for (j = colpre + 1; j <= col; j++)
				Aclp[j] = k;
			colpre = col;
		}
	}
	/* take care of empty columns at the end*/
	for (j = col + 1; j <= ncl; j++) {
		Aclp[j] = nnz;
	}
	A->assigned = 1;
	return A;
}

int spmat_set_elem(spmatrix* A, int i, int j, double Aij)		/* return -1 if (i,j) does not exist */
{
	int k;
	assert(SPMAT(A));
	assert(i < A->nrw);
	assert(j < A->ncl);
	assert(i >= 0);
	assert(j >= 0);

	for (k = A->clp[j]; k < A->clp[j + 1]; k++) {
		if (i == A->rwi[k]) {
			A->val[k] = Aij;
			return k;
		}
	}
	return -1;
}

int spmat_inc_elem(spmatrix* A, int i, int j, double delta)	/* Aij += delta, return -1 if (i,j) not found */
{
	int k;
	assert(SPMAT(A));
	assert(i < A->nrw);
	assert(j < A->ncl);
	assert(i >= 0);
	assert(j >= 0);

	for (k = A->clp[j]; k < A->clp[j + 1]; k++) {
		if (i == A->rwi[k]) {
			A->val[k] += delta;
			return k;
		}
	}
	return -1;
}

/* BLAS gemv function: y := alpha * A * x + beta * y */
void spmat_aAxby(double alpha, const spmatrix* A, const dnvector* x, double beta, dnvector* y)
{
	int i, j, m, n, *Aclp, *Arwi;
	double *Aval, *xval, *yval;

	assert(SPMAT(A));
	assert(DNVEC(x));
	assert(DNVEC(y));
	assert(x->len == A->ncl);
	assert(y->len == A->nrw);

	m = A->nrw;	n = A->ncl; Aclp = A->clp; Arwi = A->rwi;
	Aval = A->val; xval = x->val; yval = y->val;
	for (i = 0; i < m; i++)
		yval[i] *= beta;
	for (j = 0; j < n; j++){
		for (i = Aclp[j]; i < Aclp[j + 1]; i++) {
			yval[Arwi[i]] += alpha * Aval[i] * xval[j];
		}
	}
}

/* BLAS gemv function: y := alpha * A' * x + beta * y */
void spmat_aATxby(double alpha, const spmatrix* A, const dnvector* x, double beta, dnvector* y)
{
	int i, j, m, n, *Aclp, *Arwi;
	double *Aval, *xval, *yval;

	assert(SPMAT(A));
	assert(DNVEC(x));
	assert(DNVEC(y));
	assert(x->len == A->nrw);
	assert(y->len == A->ncl);

	m = A->nrw;	n = A->ncl; Aclp = A->clp; Arwi = A->rwi;
	Aval = A->val; xval = x->val; yval = y->val;
	for (j = 0; j < n; j++){
		yval[j] *= beta;
		for (i = Aclp[j]; i < Aclp[j + 1]; i++){
			yval[j] += alpha * Aval[i] * xval[Arwi[i]];
		}
	}
}

spmatrix* spmat_transpose(const spmatrix *A)
{
	int m, n, nnz, j, p, q, *ws, *Aclp, *Arwi, *Tclp, *Trwi;
	double *Aval, *Tval;
	spmatrix *T;

	assert(SPMAT(A));

	m = A->nrw; n = A->ncl; nnz = A->nnz;
	Aclp = A->clp; Arwi = A->rwi; Aval = A->val;
	T = spmat_alloc(n, m, nnz);
	if (!T) return NULL;
	
	ws = calloc(m, sizeof(int));
	if (!ws) { spmat_free(&T); return NULL; }
	Tclp = T->clp; Trwi = T->rwi; Tval = T->val;
	
	memset(ws, 0, m * sizeof(int));
	for (p = 0; p < nnz; p++) ws[Arwi[p]]++;

	Tclp[0] = 0;
	for (j = 0; j < m; j++) {
		Tclp[j + 1] = Tclp[j] + ws[j];
		ws[j] = Tclp[j];
	}
	for (j = 0; j < n; j++) {
		for (p = Aclp[j]; p < Aclp[j + 1]; p++) {
			Trwi[q = ws[Arwi[p]]++] = j;
			Tval[q] = Aval[p];
		}
	}
	free(ws);
	T->assigned = 1;
	return T;
}

spmatrix* spmat_stack_cols(const spmatrix* A, const spmatrix* B)
{
	int j, nrw, ncl, nnz, Ancl, Annz, Bncl, *ABclp;
	spmatrix* AB;

	assert(SPMAT(A));
	assert(SPMAT(B));
	assert(A->nrw == B->nrw);
	
	nrw = A->nrw;
	ncl = A->ncl + B->ncl;
	nnz = A->nnz + B->nnz;
	AB = spmat_alloc(nrw, ncl, nnz);
	if (!AB) return NULL;

	memcpy(AB->clp, A->clp, (A->ncl + 1) * sizeof(int));
	memcpy(AB->rwi, A->rwi, (A->nnz) * sizeof(int));
	memcpy(AB->val, A->val, (A->nnz) * sizeof(double));
	memcpy(AB->clp + A->ncl, B->clp, (B->ncl + 1) * sizeof(int));
	memcpy(AB->rwi + A->nnz, B->rwi, (B->nnz) * sizeof(int));
	memcpy(AB->val + A->nnz, B->val, (B->nnz) * sizeof(double));

	Ancl = A->ncl; Annz = A->nnz;  Bncl = B->ncl; ABclp = AB->clp;
	for (j = 0; j <= Bncl; j++) ABclp[Ancl + j] += Annz;
	AB->assigned = 1;
	
	return AB;
}

spmatrix* spmat_stack_rows(const spmatrix* A, const spmatrix* B)
{
	int i, j, k, nrw, ncl, nnz, Anrw, *Aclp, *Arwi, *Bclp, *Brwi, *ABclp, *ABrwi;
	double *Aval, *Bval, *ABval;
	spmatrix* AB;

	assert(SPMAT(A));
	assert(SPMAT(B));
	assert(A->ncl == B->ncl);

	nrw = A->nrw + B->nrw;
	ncl = A->ncl;			/* should have A->ncl == B->ncl */
	nnz = A->nnz + B->nnz;
	AB = spmat_alloc(nrw, ncl, nnz);
	if (!AB) return NULL;
	Aclp = A->clp; Arwi = A->rwi; Aval = A->val; Anrw = A->nrw;
	Bclp = B->clp; Brwi = B->rwi; Bval = B->val;
	ABclp = AB->clp; ABrwi = AB->rwi; ABval = AB->val;
	for (j = 0; j < ncl; j++) {
		ABclp[j] = Aclp[j] + Bclp[j];
		for (i = Aclp[j]; i < Aclp[j + 1]; i++) {
			k = Bclp[j] + i;
			ABrwi[k] = Arwi[i];
			ABval[k] = Aval[i];
		}
		for (i = Bclp[j]; i < Bclp[j + 1]; i++){
			k = Aclp[j + 1] + i;
			ABrwi[k] = Anrw + Brwi[i];
			ABval[k] = Bval[i];
		}
	}
	AB->clp[ncl] = nnz;
	AB->assigned = 1;
	return AB;
}

spmatrix* spmat_stack_symm(const spmatrix* A11, const spmatrix* A21, const spmatrix* A22)
{
	spmatrix *A = NULL, *A12 = NULL, *A1_ = NULL, *A2_ = NULL;

	assert(SPMAT(A11));
	assert(SPMAT(A21));
	assert(SPMAT(A22));
	assert(A11->nrw == A11->ncl);
	assert(A22->nrw == A22->ncl);
	assert(A11->ncl == A21->ncl);
	assert(A21->nrw == A22->nrw);
	
	A12 = spmat_transpose(A21);
	A1_ = spmat_stack_cols(A11, A12);
	A2_ = spmat_stack_cols(A21, A22);
	A = spmat_stack_rows(A1_, A2_);
	spmat_free(&A12);
	spmat_free(&A1_);
	spmat_free(&A2_);

	if (!A) return NULL;
	
	return A;
}

void spmat_print(const spmatrix* A, int showval)
{
	int i, j;
	char *s, s0[50];
	spmatrix *T;

	assert(SPMAT(A));
	s = malloc((A->ncl + 1) * sizeof(char));
	if (!s) return;
	T = spmat_transpose(A);
	if (!T) { free(s);  return; }
	printf("spmatrix size (%d, %d) with nnz = %d\n", A->nrw, A->ncl, A->nnz);
	s[A->ncl] = '\0';
	for (j = 0; j < T->ncl; j++) {
		memset(s, '.', A->ncl);
		for (i = T->clp[j]; i < T->clp[j + 1]; i++) {
			if (showval) {
				snprintf(s0, 50, "%d", (int)T->val[i]);
				s[T->rwi[i]] = s0[0];
			}
			else {
				s[T->rwi[i]] = 'X';
			}
		}
		puts(s);
	}
	free(s);
	spmat_free(&T);
}

void spmat_print2(const spmatrix *A)
{
	int i, j;
	spmatrix *T;
	int *spmat_ind = NULL;

	assert(SPMAT(A));

	T = spmat_transpose(A);
	if (T == NULL) { return; }

	spmat_ind = calloc(T->nrw, sizeof(int)); 
	if (spmat_ind == NULL) { spmat_free(&T);  return; }

	printf("size: (%d, %d)\n nnz: %d\n", A->nrw, A->ncl, A->nnz);
	for (j = 0; j < T->ncl; j++) {
		for (i = 0; i < T->nrw; i++) spmat_ind[i] = -1;
		for (i = T->clp[j]; i < T->clp[j + 1]; i++) spmat_ind[T->rwi[i]] = i;
		for (i = 0; i < T->nrw; i++) {
			if (spmat_ind[i] == -1) printf("%10s", ".");
			else printf("%10.3f", T->val[spmat_ind[i]]);
		}
		printf("\n");
	}

	free(spmat_ind);
	spmat_free(&T);
}


spmatrix *spmat_add(const spmatrix *A, const spmatrix *B) {
	spmatrix *C = NULL; /* C = A + B. */
	int C_nnz = 0;
	int Ai, Bi, j, Cx;

	assert(SPMAT(A));
	assert(SPMAT(B));
	assert(A->nrw == B->nrw);
	assert(A->ncl == B->ncl);

	/* Determine the number of nonzero elements in C. */
	for (j = 0; j < A->ncl; j++) /* Loop through the columns of A and B simultaneously. */
	{
		Bi = B->clp[j];
		for (Ai = A->clp[j]; Ai < A->clp[j + 1]; Ai++) {
			/* Count nonzero contributions from B. */
			while (Bi < B->clp[j + 1] && B->rwi[Bi] < A->rwi[Ai]) {
				C_nnz++;
				Bi++;
			}

			/* Count nonzero contributions from A + B. */
			if (Bi < B->clp[j + 1] && B->rwi[Bi] == A->rwi[Ai]) {
				if (B->val[Bi] + A->val[Ai] != 0) {
					C_nnz++;
				}
				Bi++;
			}
			else {
				/* Count nonzero contributions from A. */
				C_nnz++;
			}
		}

		/* Finish counting contributions from B (when the jth column of B has more nonzero elements than the jth column of A). */
		C_nnz = C_nnz + B->clp[j + 1] - Bi;
	}
	
	/* Allocate space for C. */
	C = spmat_alloc(A->nrw, A->ncl, C_nnz);

	/* Set C = A + B. */
	Cx = 0; /* Index for C->val and C->rwi. */
	for (j = 0; j < A->ncl; j++) /* Loop through the columns of A, B, and C simultaneously. */
	{
		C->clp[j] = Cx;
		Bi = B->clp[j];
		for (Ai = A->clp[j]; Ai < A->clp[j + 1]; Ai++) {
			/* Set nonzero contribution from B. */
			while (Bi < B->clp[j + 1] && B->rwi[Bi] < A->rwi[Ai]) {
				C->rwi[Cx] = B->rwi[Bi];
				C->val[Cx] = B->val[Bi];
				Cx++;
				Bi++;
			}

			/* Compute nonzero contribution from A + B. */
			if (Bi < B->clp[j + 1] && B->rwi[Bi] == A->rwi[Ai]) {
				if (B->val[Bi] + A->val[Ai] != 0) {
					C->rwi[Cx] = A->rwi[Ai];
					C->val[Cx] = A->val[Ai] + B->val[Bi];
					Cx++;
				}
				Bi++;
			}
			else {
				/* Set nonzero contribution from A. */
				C->rwi[Cx] = A->rwi[Ai];
				C->val[Cx] = A->val[Ai];
				Cx++;
			}
		}

		/* Finish contributions from B. */
		while (Bi < B->clp[j + 1]) {
			C->rwi[Cx] = B->rwi[Bi];
			C->val[Cx] = B->val[Bi];
			Cx++;
			Bi++;
		}
	}

	C->clp[C->ncl] = Cx;
	C->assigned = 1;
	
	return C;
}

void spmat_add_diagonal(spmatrix **pA, const dnvector *v) {
	spmatrix *A_updated = NULL;
	spmatrix *V = NULL;

	assert((*pA)->ncl == (*pA)->nrw);
	assert((*pA)->ncl == v->len);
	
	V = spmat_diag(v);
	A_updated = spmat_add(*pA, V);
	
	spmat_free(pA);
	*pA = A_updated;
	spmat_free(&V);
}

void spmat_pcprint(const spmatrix *A) {
	int i, j;
	int first_in_column;

	assert(SPMAT(A));

	printf("size: (%d, %d)\n nnz: %d\n\n", A->nrw, A->ncl, A->nnz);

	printf("%5s : %5s : %s\n", "clp", "rwi", "val");
	for (j = 0; j < A->ncl; j++) {
		first_in_column = 1;
		for (i = A->clp[j]; i < A->clp[j + 1]; i++) {
			if (first_in_column) {
				printf("%5d : %5d : %.10f\n", A->clp[j], A->rwi[i], A->val[i]);
				first_in_column = 0;
			}
			else
				printf("%5s : %5d : %.10f\n", "", A->rwi[i], A->val[i]);
		}
	}
	printf("%5d : %5s : %s\n\n", A->clp[A->ncl], "", "");

}

/* Random sparse matrix of size (nrw, ncl).  density approximates the true density. */
spmatrix *spmat_rand(int nrw, int ncl, double density)
{
	spmatrix *A = NULL;
	spmat_list *splist = NULL;
	int nnz;
	int i, j, k;


	assert(nrw >= 0);
	assert(ncl >= 0);
	assert(0 <= density);

	nnz = (int)floor(nrw * ncl * density);
	splist = spmat_list_alloc(nnz);
	for (k = 0; k < nnz; k++) {
		i = rand() % nrw;
		j = rand() % ncl;
		spmat_list_add(splist, i, j, rand() / (double)RAND_MAX);
	}

	nnz = spmat_list_sort_dupl(splist);
	A = spmat_alloc_splist(nrw, ncl, splist);
	assert(A);

	spmat_list_free(&splist);

	return A;
}

void spmat_scale(spmatrix * A, double s)
{
	int i;

	for (i = 0; i < A->nnz; i++)
		A->val[i] *= s;
}