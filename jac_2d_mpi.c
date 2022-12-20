#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define N (2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 + 2)
#define TAG_0 123
#define TAG_1 345
float maxeps = 1e-8;
int itmax = 100;
int it, i, j, k;
float eps, sum;

void relax();
void resid();
void init();
void verify();

MPI_Request req[4];
MPI_Status status[4];
int myid, nproc;
int start_row, end_row, stride;
float **A, **B;

int main(int an, char **as)
{
        MPI_Init(&an, &as);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);

        start_row = (N * myid) / nproc;
        end_row = (N * (myid + 1)) / nproc - 1;
        stride = end_row - start_row + 1;

        A = (float**) malloc((stride + 2) * sizeof(float*));
        B = (float**) malloc((stride) * sizeof(float*));

        for (i = 0; i < stride + 2; i++) {
                A[i] = (float*) malloc(N * sizeof(float));
        }

        for (i = 0; i < stride; i++) {
                B[i] = (float*) malloc(N * sizeof(float));
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();

        init();
        for (it = 1; it <= itmax; ++it) {
                resid();
                relax();
                if (myid == 0) {
                        printf("it=%4i, eps=%f\n", it , eps);
                }
                if (eps < maxeps) break;
        }
        verify();

        MPI_Barrier(MPI_COMM_WORLD);

        if (myid == 0) {
                double end = MPI_Wtime() - start;
                printf("%d %lf\n", nproc, end);
        }

        MPI_Finalize();

        return 0;
}

void init()
{
        for (i = 1; i <= stride; ++i) {
                for (j = 0; j <= N - 1; ++j) {
                        B[i - 1][j] = 1. + start_row + i + j - 1;
                        A[i][j] = 0.;
                }
        }
}

void relax()
{
        if (nproc > 1) {
                if (myid != 0) {
                        MPI_Irecv(&A[0][0], N, MPI_FLOAT, myid - 1, TAG_0, MPI_COMM_WORLD, &req[0]);
                }
                if (myid != nproc - 1) {
                        MPI_Isend(&A[stride][0], N, MPI_FLOAT, myid + 1, TAG_0, MPI_COMM_WORLD, &req[2]);
                }
                if (myid != nproc - 1) {
                        MPI_Irecv(&A[stride + 1][0], N, MPI_FLOAT, myid + 1, TAG_1, MPI_COMM_WORLD, &req[3]);
                }
                if (myid != 0) {
                        MPI_Isend(&A[1][0], N, MPI_FLOAT, myid - 1, TAG_1, MPI_COMM_WORLD, &req[1]);
                }
                int ll = 4, shift = 0;
                if (myid == 0) {
                        ll = 2;
                        shift = 2;
                }
                if (myid == nproc - 1) {
                        ll = 2;
                }
                MPI_Waitall(ll, &req[shift], &status[0]);
        }

        for (i = 1; i<= stride; ++i) {
                for (j = 1; j<= N - 2; ++j) {
                        B[i - 1][j] = (A[i - 1][j] + A[i+1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
                }
        }
}

void resid()
{
        float e = maxeps;
        for (i = 1; i<= stride; ++i) {
                for (j = 1; j <= N - 2; ++j) {
                        if (((i == 1) && (myid == 0)) || ((i == stride) && (myid == nproc - 1))) {
                                continue;
                        }
                        e = Max(e, fabs(A[i][j] - B[i - 1][j]));
                        A[i][j] = B[i - 1][j];
                }
        }
        MPI_Allreduce(&e, &eps, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
}

void verify()
{
        float s = 0.;
        for (i = 1; i <= stride; ++i) {
                for (j = 0; j <= N - 1; ++j)
                {
                        s += A[i - 1][j] * (i + start_row) * (j + 1) / (N * N);
                }
        }
        MPI_Reduce(&s, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (myid == 0) {
                printf("s = %f\n", sum);
        }
}