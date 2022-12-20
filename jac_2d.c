#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2)
float   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
float eps;
float A [N][N],  B [N][N];

void relax();
void resid();
void init();
void verify(); 

int main(int an, char **as)
{
	int it;
	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	verify();
	return 0;
}

void init()
{ 
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
		else A[i][j]= ( 1. + i + j ) ;
	}
} 

void relax()
{
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{
		B[i][j]=(A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.;
	}
}

void resid()
{ 
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{
		float e;
		e = fabs(A[i][j] - B[i][j]);         
		A[i][j] = B[i][j]; 
		eps = Max(eps,e);
	}
}

void verify()
{
	float s;
	s=0.;
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		s=s+A[i][j]*(i+1)*(j+1)/(N*N);
	}
	printf("  S = %f\n",s);
}
