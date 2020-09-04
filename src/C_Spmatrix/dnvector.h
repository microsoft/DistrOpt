/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */

#ifndef DNVECTOR_H
#define DNVECTOR_H

/* DEVEC(v) is true if v satisfies: allocated && (own memory || pointed memory exist) */
#define DNVEC(v) (v && (v->ownmem || *(v->ppm)))

typedef struct dense_vector
{
	int len;
	int ownmem;		/* 1 if it owns allocated memory, 0 if does not own memory */
	double* val;	/* pointer to allocated memory, either own or from other v */
	void** ppm;		/* pointer to (pointer to a dnvector which has own memory) */
} dnvector;

/* Note: only declare pointers to dnvector and initialize/free them with the following functions! */
dnvector* dnvec_alloc(int len);
dnvector* dnvec_alloc_init(int len, double *val);
dnvector* dnvec_alias(dnvector **pp, int start, int length);
void      dnvec_free(dnvector** pp);

dnvector *dnvec_rand(int len);

/* simple vector operations */
void	dnvec_copy(dnvector* u, const dnvector* v);			/* u copy from v       */
void    dnvec_set_elem(dnvector* v, int i, double x);   /* v[i] = x            */
void    dnvec_set_zero(dnvector* v);					/* v[i] = 0 for all i  */
void    dnvec_set_const(dnvector* v, double c);			/* v[i] = c for all i  */
void    dnvec_add_const(dnvector* v, double c);			/* v[i] += c for all i */
void    dnvec_scale(dnvector* v, double s);
double  dnvec_sum(const dnvector* v);
double  dnvec_dot(const dnvector* u, const dnvector* v);
double  dnvec_norm_1(const dnvector* v);
double  dnvec_norm_2(const dnvector* v);
double  dnvec_norm_inf(const dnvector* v);
void	dnvec_elem_abs(const dnvector* x, dnvector* y);		/* y = |x|        */
void	dnvec_elem_inv(const dnvector* x, dnvector* y);		/* y = 1./x       */
void	dnvec_elem_exp(const dnvector* x, dnvector* y);		/* y = exp(x)     */
void	dnvec_elem_log(const dnvector* x, dnvector* y);		/* y = log(x)     */
void	dnvec_elem_expm1(const dnvector* x, dnvector* y);		/* y = exp(x) - 1 */
void	dnvec_elem_log1p(const dnvector* x, dnvector* y);		/* y = log(1 + x) */
void	dnvec_elem_sqrt(const dnvector* x, dnvector* y);		/* y = sqrt(x)    */
void    dnvec_axpy(double alpha, const dnvector* x, dnvector* y);				   /* y := alpha * x + y */
void    dnvec_elem_op(char op, const dnvector* x, const dnvector* y, dnvector* z);	   /* z = x (+,-,*,/) y */
double dnvec_mean(const dnvector* x);
double dnvec_min(const dnvector* x);
double dnvec_max(const dnvector* x);
void dnvec_print(const dnvector *x);

#endif