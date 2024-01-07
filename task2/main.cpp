#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <signal.h>
#include <mpi-ext.h>
#include <setjmp.h>
#include <string.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

constexpr int N = 4096;
constexpr int TAG_0 = 123;
constexpr int TAG_1 = 345;

constexpr int itmax = 100;

const char cpA[] = "checkpointA";
const char cpB[] = "checkpointB";

int id_to_kill = 1;

float maxeps = 1e-8;

int it, i, j, k;
float eps, sum;

jmp_buf jbuf;
MPI_Comm global_comm = MPI_COMM_WORLD;

void relax();
void resid();
void allocate();
void deallocate();
void init();
void verify();

void save_checkpoint();
void load_checkpoint();
static void error_handler(MPI_Comm *comm, int *err, ...);

MPI_Request req[4];
MPI_Status status[4];
int myid, nproc;
int start_row, end_row, stride;
float **A, **B;

void printA() {
        if (myid != 0) {
                return;
        }
        for (int l = 0; l < stride + 2; ++l) {
                for (int h = 0; h < N; ++h) {
                        printf("%.3f ", A[l][h]);
                        fflush(stdout);
                }
                printf("\n");
                fflush(stdout);
        }
}

int main(int an, char **as)
{
        MPI_Init(&an, &as);
        MPI_Comm_rank(global_comm, &myid);
        MPI_Comm_size(global_comm, &nproc);

        MPI_Errhandler errh;

        MPI_Comm_create_errhandler(error_handler, &errh);
        MPI_Comm_set_errhandler(global_comm, errh);

        MPI_Barrier(global_comm);

        allocate();

        double start = MPI_Wtime();

        init();
        save_checkpoint();
        setjmp(jbuf);
        for (it = 1; it <= itmax; ++it) {
                load_checkpoint();
                resid();
                if (myid == id_to_kill) {
                        printf("Process %d killed\n", myid);
                        fflush(stdout);
                        raise(SIGKILL);
                }
                relax();
                save_checkpoint();
                if (myid == 0) {
                        printf("it=%4i, eps=%f\n", it , eps);
                }
                if (eps < maxeps) break;
        }
        load_checkpoint();
        verify();

        MPI_Barrier(global_comm);

        if (myid == 0) {
                double end = MPI_Wtime() - start;
                printf("%d %lf\n", nproc, end);
        }

        MPI_Finalize();

        return 0;
}

void allocate() {
        start_row = (N * myid) / nproc;
        end_row = (N * (myid + 1)) / nproc - 1;
        stride = end_row - start_row + 1;

        A = (float**) malloc((stride + 2) * sizeof(float*));
        B = (float**) malloc((stride) * sizeof(float*));

        for (i = 0; i < stride + 2; i++) {
                A[i] = (float*) malloc(N * sizeof(float));
                memset(A[i], 0, N);
        }

        for (i = 0; i < stride; i++) {
                B[i] = (float*) malloc(N * sizeof(float));
        }

        MPI_Barrier(global_comm);
}

void deallocate() {
        for (i = 0; i < stride + 2; i++) {
                free(A[i]);
        }
        for (i = 0; i < stride; i++) {
                free(B[i]);
        }
        free(A);
        free(B);
}

void init()
{
        for (i = 1; i <= stride; ++i) {
                for (j = 0; j < N; ++j) {
                        B[i - 1][j] = 1. + start_row + i + j - 1;
                        A[i][j] = 0.;
                }
        }
}

void relax()
{
        if (nproc > 1) {
                if (myid != 0) {
                        MPI_Irecv(&A[0][0], N, MPI_FLOAT, myid - 1, TAG_0, global_comm, &req[0]);
                }
                if (myid != nproc - 1) {
                        MPI_Isend(&A[stride][0], N, MPI_FLOAT, myid + 1, TAG_0, global_comm, &req[2]);
                }
                if (myid != nproc - 1) {
                        MPI_Irecv(&A[stride + 1][0], N, MPI_FLOAT, myid + 1, TAG_1, global_comm, &req[3]);
                }
                if (myid != 0) {
                        MPI_Isend(&A[1][0], N, MPI_FLOAT, myid - 1, TAG_1, global_comm, &req[1]);
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

        for (i = 1; i <= stride; ++i) {
                for (j = 1; j<= N - 2; ++j) {
                        B[i - 1][j] = (A[i - 1][j] + A[i+1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
                }
        }
}

void resid()
{
        float e = maxeps;
        for (i = 1; i <= stride; ++i) {
                for (j = 1; j <= N - 2; ++j) {
                        if (((i == 1) && (myid == 0)) || ((i == stride) && (myid == nproc - 1))) {
                                continue;
                        }
                        e = Max(e, fabs(A[i][j] - B[i - 1][j]));
                        A[i][j] = B[i - 1][j];
                }
        }
        MPI_Allreduce(&e, &eps, 1, MPI_FLOAT, MPI_MAX, global_comm);
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
        MPI_Reduce(&s, &sum, 1, MPI_FLOAT, MPI_SUM, 0, global_comm);

        if (myid == 0) {
                printf("s = %f\n", sum);
        }
}

void save_checkpoint()
{
        MPI_File fileA;
        MPI_File_open(global_comm, cpA, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileA);
        if (myid != 0) {
                i = start_row - 1;
                MPI_File_write_at(fileA, sizeof(MPI_FLOAT) * N * i, A[0], N, MPI_FLOAT, MPI_STATUS_IGNORE);
        }
        for (i = start_row; i <= end_row; ++i) {
                MPI_File_write_at(fileA, sizeof(MPI_FLOAT) * N * i, A[i - start_row + 1], N, MPI_FLOAT, MPI_STATUS_IGNORE);
        }
        MPI_File_close(&fileA);
        if (myid != nproc - 1) {
                i = end_row + 1;
                MPI_File_write_at(fileA, sizeof(MPI_FLOAT) * N * i, A[stride + 1], N, MPI_FLOAT, MPI_STATUS_IGNORE);
        }

        MPI_File fileB;
        MPI_File_open(global_comm, cpB, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileB);
        for (i = start_row; i <= end_row; ++i) {
                MPI_File_write_at(fileB, sizeof(MPI_FLOAT) * N * i, B[i - start_row], N, MPI_FLOAT, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(global_comm);
        MPI_File_close(&fileB);
}

void load_checkpoint()
{
        MPI_File fileA;
        MPI_File_open(global_comm, cpA, MPI_MODE_RDONLY, MPI_INFO_NULL, &fileA);
        if (myid != 0) {
                i = start_row - 1;
                MPI_File_read_at(fileA, sizeof(MPI_FLOAT) * N * i, A[0], N, MPI_FLOAT, MPI_STATUS_IGNORE);
        }
        for (i = start_row; i <= end_row; ++i) {
                MPI_File_read_at(fileA, sizeof(MPI_FLOAT) * N * i, A[i - start_row + 1], N, MPI_FLOAT, MPI_STATUS_IGNORE);
        }
        MPI_File_close(&fileA);
        if (myid != nproc - 1) {
                i = end_row + 1;
                MPI_File_read_at(fileA, sizeof(MPI_FLOAT) * N * i, A[stride + 1], N, MPI_FLOAT, MPI_STATUS_IGNORE);
        }

        MPI_File fileB;
        MPI_File_open(global_comm, cpB, MPI_MODE_RDONLY, MPI_INFO_NULL, &fileB);
        for (i = start_row; i <= end_row; ++i) {
                MPI_File_read_at(fileB, sizeof(MPI_FLOAT) * N * i, B[i - start_row], N, MPI_FLOAT, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(global_comm);
        MPI_File_close(&fileB);
}

static void error_handler(MPI_Comm *comm, int *err, ...)
{
        int len;
        char errstr[MPI_MAX_ERROR_STRING];
        MPI_Error_string(*err, errstr, &len);
        printf("Process %d, got error: %.*s\n", myid, len, errstr);
        fflush(stdout);
        
        id_to_kill = -1;
        deallocate();

        fflush(stdout);
        
        MPIX_Comm_shrink(*comm, &global_comm);
        MPI_Comm_rank(global_comm, &myid);
        MPI_Comm_size(global_comm, &nproc);

        allocate();

        fflush(stdout);
        
        MPI_Barrier(global_comm);
        
        longjmp(jbuf, 0);
}
