/* Copyright (c) Microsoft Corporation.
   Licensed under the MIT License. */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "spmatrix.h"
#include "dnvector.h"

int main()
{
	spmatrix *A = NULL;
	spmatrix *B = NULL;
	spmatrix *C = NULL;
	spmatrix *D = NULL;
	spmatrix *E = NULL;
	spmatrix *Z = NULL;
	dnvector *x = NULL;
	dnvector *y = NULL;
	dnvector *z = NULL;

	int m = 5, m2 = 4;
	int n = 5, n2 = 3;


	/* Testing spmat_row and spmat_col */
	printf("\n\n\n\nTesting spmat_row and spmat_col.\n\n\n\n");
	dnvec_free(&x); x = dnvec_alloc(n);
	spmat_free(&A); A = spmat_row(x);
	spmat_free(&B); B = spmat_col(x);

	dnvec_print(x);
	spmat_print2(A);
	spmat_print2(B);

	dnvec_free(&x); x = dnvec_alloc(0);
	spmat_free(&A); A = spmat_row(x);
	spmat_free(&B); B = spmat_col(x);

	dnvec_print(x);
	spmat_print2(A);
	spmat_print2(B);



	/* Testing transpose */
	printf("\n\n\n\nTesting add.\n\n\n\n");
	spmat_free(&A); A = spmat_rand(n, n, .5);
	spmat_free(&B); B = spmat_rand(n, n, 0.1);
	spmat_free(&D); D = spmat_eye(n);
	spmat_free(&Z); Z = spmat_zero(n, n);
	spmat_free(&E); E = spmat_zero(0, m);

	printf("A\n");
	spmat_print2(A);

	printf("B\n");
	spmat_print2(B);

	printf("D\n");
	spmat_print2(D);

	printf("Z\n");
	spmat_print2(Z);

	printf("E\n");
	spmat_print2(E);




	spmat_free(&C); C = spmat_add(A, A);
	printf("A+A\n");
	spmat_print2(C);

	spmat_free(&C); C = spmat_add(A, B);
	printf("A+B\n");
	spmat_print2(C);

	spmat_free(&C); C = spmat_add(D, A);
	printf("D+A\n");
	spmat_print2(C);

	spmat_free(&C); C = spmat_add(A, D);
	printf("A+D\n");
	spmat_print2(C);

	spmat_free(&C); C = spmat_add(A, Z);
	printf("A+Z\n");
	spmat_print2(C);

	spmat_free(&C); C = spmat_add(Z, A);
	printf("Z+A\n");
	spmat_print2(C);

	spmat_free(&B);  B = spmat_add(A, Z);
	spmat_scale(B, -1);
	spmat_free(&C); C = spmat_add(A, B);
	printf("A-A\n");
	spmat_print2(C);

	printf("E+E\n");
	spmat_free(&C); C = spmat_add(E, E);
	spmat_print2(C);




	printf("Creating a random sparse matrix A.\n");
	spmat_free(&A); A = spmat_rand(m, m, .5);
	spmat_print2(A);

	printf("Creating a diagonal sparse matrix B.\n");
	dnvec_free(&x); x = dnvec_alloc(m); dnvec_set_const(x, 3);
	spmat_free(&B); B = spmat_diag(x);
	spmat_print2(B);

	printf("Setting C = A + B.\n");
	spmat_free(&C); C = spmat_add(A, B);
	spmat_print2(C);

	printf("Setting D = C - A\n");
	spmat_scale(A, -1);
	spmat_free(&D); D = spmat_add(C, A);
	spmat_pcprint(D);
	spmat_print2(D);

	/* Testing transpose */
	printf("\n\n\n\nTesting transpose.\n\n\n\n");
	spmat_free(&A); A = spmat_rand(m, n, .5);
	spmat_free(&B); B = spmat_transpose(A);

	printf("A\n");
	spmat_print2(A);

	printf("AT\n");
	spmat_print2(B);

	spmat_free(&Z); Z = spmat_zero(m, n);
	spmat_free(&B); B = spmat_transpose(Z);

	printf("Z\n");
	spmat_print2(Z);

	printf("ZT\n");
	spmat_print2(B);

	spmat_free(&E); E = spmat_zero(0, n);
	spmat_free(&B); B = spmat_transpose(E);

	printf("E\n");
	spmat_print2(E);

	printf("ET\n");
	spmat_print2(B);


	spmat_free(&E); E = spmat_zero(m, 0);
	spmat_free(&B); B = spmat_transpose(E);

	printf("E\n");
	spmat_print2(E);

	printf("ET\n");
	spmat_print2(B);

	spmat_free(&E); E = spmat_zero(0, 0);
	spmat_free(&B); B = spmat_transpose(E);

	printf("E\n");
	spmat_print2(E);

	printf("ET\n");
	spmat_print2(B);


	/* Testing stack_cols */
	printf("\n\n\n\nTesting stack_cols.\n\n\n\n");
	spmat_free(&A); A = spmat_rand(m, n, .5);
	spmat_free(&B); B = spmat_rand(m, n2, 0.2);
	spmat_free(&E); E = spmat_zero(m, 0);
	spmat_free(&Z); Z = spmat_zero(m, n2);

	printf("A\n");
	spmat_print2(A);

	printf("B\n");
	spmat_print2(B);

	printf("E\n");
	spmat_print2(E);

	printf("Z\n");
	spmat_print2(Z);

	printf("[A, B]\n");
	spmat_free(&C); C = spmat_stack_cols(A, B);
	spmat_print2(C);

	printf("[A, Z]\n");
	spmat_free(&C); C = spmat_stack_cols(A, Z);
	spmat_print2(C);

	printf("[Z, A]\n");
	spmat_free(&C); C = spmat_stack_cols(Z, A);
	spmat_print2(C);

	printf("[A, E]\n");
	spmat_free(&C); C = spmat_stack_cols(A, E);
	spmat_print2(C);

	printf("[E, A]\n");
	spmat_free(&C); C = spmat_stack_cols(E, A);
	spmat_print2(C);

	/* Testing stack_rows */
	printf("\n\n\n\nTesting stack_rows.\n\n\n\n");
	spmat_free(&A); A = spmat_rand(m, n, .5);
	spmat_free(&B); B = spmat_rand(m2, n, 0.2);
	spmat_free(&E); E = spmat_zero(0, n);
	spmat_free(&Z); Z = spmat_zero(m2, n);

	printf("A\n");
	spmat_print2(A);

	printf("B\n");
	spmat_print2(B);

	printf("E\n");
	spmat_print2(E);

	printf("Z\n");
	spmat_print2(Z);

	printf("[A; B]\n");
	spmat_free(&C); C = spmat_stack_rows(A, B);
	spmat_print2(C);

	printf("[A; Z]\n");
	spmat_free(&C); C = spmat_stack_rows(A, Z);
	spmat_print2(C);

	printf("[Z; A]\n");
	spmat_free(&C); C = spmat_stack_rows(Z, A);
	spmat_print2(C);

	printf("[A; E]\n");
	spmat_free(&C); C = spmat_stack_rows(A, E);
	spmat_print2(C);

	printf("[E; A]\n");
	spmat_free(&C); C = spmat_stack_rows(E, A);
	spmat_print2(C);

	/* Testing stack_symm */
	printf("\n\n\n\nTesting stack_symm.\n\n\n\n");
	spmat_free(&A); A = spmat_rand(n, n, .5);
	spmat_free(&B); B = spmat_zero(m, n);
	spmat_free(&C); C = spmat_rand(m, m, 0.5);
	spmat_free(&D); D = spmat_stack_symm(A, B, C);

	printf("A\n");
	spmat_print2(A);

	printf("B\n");
	spmat_print2(B);

	printf("C\n");
	spmat_print2(C);

	printf("[A, BT; B, C]\n");
	spmat_print2(D);

	spmat_free(&A); A = spmat_rand(0, 0, .5);
	spmat_free(&B); B = spmat_zero(m, 0);
	spmat_free(&C); C = spmat_rand(m, m, 0.5);
	spmat_free(&D); D = spmat_stack_symm(A, B, C);

	printf("A\n");
	spmat_print2(A);

	printf("B\n");
	spmat_print2(B);

	printf("C\n");
	spmat_print2(C);

	printf("[A, BT; B, C]\n");
	spmat_print2(D);

	spmat_free(&A); A = spmat_rand(n, n, .5);
	spmat_free(&B); B = spmat_zero(0, n);
	spmat_free(&C); C = spmat_rand(0, 0, 0.5);
	spmat_free(&D); D = spmat_stack_symm(A, B, C);

	printf("A\n");
	spmat_print2(A);

	printf("B\n");
	spmat_print2(B);

	printf("C\n");
	spmat_print2(C);

	printf("[A, BT; B, C]\n");
	spmat_print2(D);

	spmat_free(&A);
	spmat_free(&B);
	spmat_free(&C);
	spmat_free(&D);
	spmat_free(&E);
	spmat_free(&Z);
	dnvec_free(&x);
	dnvec_free(&y);
	dnvec_free(&z);

	return 0;
}