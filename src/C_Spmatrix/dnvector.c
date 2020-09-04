/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "dnvector.h"

/* Need to change all assert() statements into specific error messages and call exit() */
#define INF 1.0e300

dnvector* dnvec_alloc(int len)
{
	dnvector* v = malloc(sizeof(dnvector));
	if (!v) return NULL;
	v->len = len;
	v->val = calloc(len, sizeof(double));
	if (!v->val) { free(v); return NULL; }
	v->ownmem = 1;
	v->ppm = NULL;
	return v;
}

dnvector* dnvec_alloc_init(int len, double *val)
{
	dnvector* v = dnvec_alloc(len);
	if (!v) return NULL;

	memcpy(v->val, val, len * sizeof(double));
	return v;
}

dnvector* dnvec_alias(dnvector **pp, int start, int len)
{
	dnvector *v;
	if (!pp || !DNVEC((*pp)) || start + len > (*pp)->len) return NULL;
	v = calloc(1, sizeof(dnvector));
	if (!v) return NULL;
	v->len = len;
	v->ownmem = 0;
	v->ppm = (*pp)->ownmem ? (void**)pp : (*pp)->ppm;
	v->val = (*pp)->val + start;
	return v;
}

void dnvec_free(dnvector **pp)
{
	if (!(*pp)) return;
	if ((*pp)->ownmem) free((*pp)->val);
	free(*pp);
	*pp = NULL;
}

dnvector * dnvec_rand(int len)
{
	dnvector *random_vector = NULL;
	int i;

	assert(len >= 0);

	random_vector = dnvec_alloc(len);
	if (random_vector == NULL) return NULL;
	for (i = 0; i < len; i++)
		random_vector->val[i] = rand() / (double)RAND_MAX;

	return random_vector;
}

void dnvec_copy(dnvector* u, const dnvector* v)		/* u := v */
{
	assert(DNVEC(u) && DNVEC(v) && u->len == v->len);
	memcpy(u->val, v->val, u->len*sizeof(double));
}
void dnvec_set_elem(dnvector* v, int i, double x)   /* v[i] = x */
{
	assert(DNVEC(v) && i < v->len && i >= 0);
	v->val[i] = x;
}

void dnvec_set_zero(dnvector *v)
{
	assert(DNVEC(v));
	memset(v->val, 0, v->len*sizeof(double));
}

void dnvec_set_const(dnvector *v, double c)
{
	double *vval;
	int i, len;
	assert(DNVEC(v));
	len = v->len;  vval = v->val;
	for (i = 0; i < len; i++)
		vval[i] = c;
}

void dnvec_add_const(dnvector *v, double c)
{
	double *vval;
	int i, len;
	assert(DNVEC(v));
	len = v->len;  vval = v->val;
	for (i = 0; i < len; i++)
		vval[i] += c;
}

void dnvec_scale(dnvector *v, double s)
{
	double *vval;
	int i, len;
	assert(DNVEC(v));
	len = v->len;  vval = v->val;
	for (i = 0; i < len; i++)
		vval[i] *= s;
}

double dnvec_sum(const dnvector *v)
{
	double *vval, sum;
	int i, len;
	assert(DNVEC(v));
	len = v->len; vval = v->val;
	sum = 0;
	for (i = 0; i < len; i++)
		sum += vval[i];
	return sum;
}

double dnvec_dot(const dnvector *u, const dnvector *v)
{
	double *uval, *vval, sum;
	int i, len;
	assert(DNVEC(u) && DNVEC(v) && u->len == v->len);
	len = u->len; uval = u->val; vval = v->val;
	sum = 0;
	for (i = 0; i < len; i++)
		sum += uval[i] * vval[i];
	return sum;
}

double dnvec_norm_1(const dnvector *v)
{
	double *vval, sumabs;
	int i, len;
	assert(DNVEC(v));
	len = v->len; vval = v->val;
	sumabs = 0;
	for (i = 0; i < len; i++)
		sumabs += fabs(vval[i]);
	return sumabs;
}

double dnvec_norm_2(const dnvector *v)
{
	double norm2 = dnvec_dot(v, v);
	return sqrt(norm2);
}

double dnvec_norm_inf(const dnvector *v)
{
	double *vval, valabs, maxabs;
	int i, len;
	assert(DNVEC(v));
	len = v->len; vval = v->val;
	maxabs = 0;
	for (i = 0; i < len; i++) {
		valabs = fabs(vval[i]);
		if (valabs>maxabs) maxabs = valabs;
	}
	return maxabs;
}

void dnvec_elem_abs(const dnvector* x, dnvector* y)	/* y = |x| */
{
	double *xval, *yval;
	int i, len;
	assert(DNVEC(x) && DNVEC(y) && x->len == y->len);
	len = x->len; xval = x->val; yval = y->val;
	for (i = 0; i < len; i++)
		yval[i] = fabs(xval[i]);
}

void dnvec_elem_inv(const dnvector* x, dnvector* y)	/* y = 1./x  */
{
	double *xval, *yval;
	int i, len;
	assert(DNVEC(x) && DNVEC(y) && x->len == y->len);
	len = x->len; xval = x->val; yval = y->val; 
	for (i = 0; i < len; i++)
		yval[i] = 1 / xval[i];
}

void dnvec_elem_exp(const dnvector* x, dnvector* y)	/* y = exp(x) */
{
	double *xval, *yval;
	int i, len;
	assert(DNVEC(x) && DNVEC(y) && x->len == y->len);
	len = x->len; xval = x->val; yval = y->val;
	for (i = 0; i < len; i++)
		yval[i] = exp(xval[i]);
}

void dnvec_elem_log(const dnvector* x, dnvector* y)	/* y = log(x) */
{
	double *xval, *yval;
	int i, len;
	assert(DNVEC(x) && DNVEC(y) && x->len == y->len);
	len = x->len; xval = x->val; yval = y->val;
	for (i = 0; i < len; i++)
		yval[i] = log(xval[i]);
}

void dnvec_elem_expm1(const dnvector* x, dnvector* y)	/* y = exp(x) - 1 */
{
	double *xval, *yval;
	int i, len;
	assert(DNVEC(x) && DNVEC(y) && x->len == y->len);
	len = x->len; xval = x->val; yval = y->val;
	for (i = 0; i < len; i++)
		yval[i] = expm1(xval[i]);
}

void dnvec_elem_log1p(const dnvector* x, dnvector* y)	/* y = log(1+x) */
{
	double *xval, *yval;
	int i, len;
	assert(DNVEC(x) && DNVEC(y) && x->len == y->len);
	len = x->len; xval = x->val; yval = y->val;
	for (i = 0; i < len; i++)
		yval[i] = log1p(xval[i]);
}

void dnvec_elem_sqrt(const dnvector* x, dnvector* y)	/* y = sqrt(x) */
{
	double *xval, *yval;
	int i, len;
	assert(DNVEC(x) && DNVEC(y) && x->len == y->len);
	len = x->len; xval = x->val; yval = y->val;
	for (i = 0; i < len; i++)
		yval[i] = sqrt(xval[i]);
}

void dnvec_axpy(double alpha, const dnvector* x, dnvector* y)		/* y := alpha * x + y */
{
	double *xval, *yval;
	int i, len;
	assert(DNVEC(x) && DNVEC(y) && x->len == y->len);
	len = x->len; xval = x->val; yval = y->val;
	for (i = 0; i < len; i++)
		yval[i] += alpha * xval[i];
}

void dnvec_elem_op(char op, const dnvector* x, const dnvector* y, dnvector* z)	/* z = x (+,-,*,/) y */
{
	double *xval, *yval, *zval;
	int i, len;
	assert(op == '+' || op == '-' || op == '*' || op == '/');
	assert(DNVEC(x) && DNVEC(y) && DNVEC(z) && x->len == y->len && y->len == z->len);
	len = x->len; xval = x->val; yval = y->val; zval = z->val;
	switch (op) {
	case '+':
		for (i = 0; i < len; i++) zval[i] = xval[i] + yval[i];
		break;
	case '-':
		for (i = 0; i < len; i++) zval[i] = xval[i] - yval[i];
		break;
	case '*':
		for (i = 0; i < len; i++) zval[i] = xval[i] * yval[i];
		break;
	case '/':
		for (i = 0; i < len; i++) zval[i] = xval[i] / yval[i];
		break;
	}
}

double dnvec_mean(const dnvector* x)
{
	assert(DNVEC(x) && x->len > 0);
	return (dnvec_sum(x) / x->len);
}

double dnvec_min(const dnvector* v)
{
	double *vval, min;
	int i, len;
	assert(DNVEC(v) && v->len > 0);
	
	len = v->len; vval = v->val;
	min = INF; /* CHECK THIS*/
	for (i = 0; i < len; i++) {
		if (vval[i] < min)
			min = vval[i];
	}

	return min;
}


double dnvec_max(const dnvector* v)
{
	double *vval, max;
	int i, len;
	assert(DNVEC(v) && v->len > 0);

	len = v->len; vval = v->val;
	max = -INF; /* CHECK THIS*/
	for (i = 0; i < len; i++) {
		if (vval[i] > max)
			max = vval[i];
	}

	return max;
}

void dnvec_print(const dnvector *x) {
	int i;

	printf("size: %d\n", x->len);

	for (i = 0; i < x->len; i++)
		printf("%f\n", x->val[i]);
	printf("\n");
}