#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>

#include <mex.h>
#include <matrix.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double *Y, *D, *R, *T, *P;  /*input*/
  double *L;  /*output*/

  double R11, R12, R13, R21, R22, R23, R31, R32, R33;
  double T1, T2, T3;
  double Y1, Y2, Y3;
  double Z1, Z2, Z3;
  double b1, b2, b3;
  double u, v;
  int    ui, vi;
  int    prep;
  double th, phi;
  double D1;
  double w, h, wfov;
  double a;

  double fov = 3.141592653589793;

  int M, N;
  int PM, PN;
  int ii, jj;

  /*read inputs*/
  Y = mxGetPr(prhs[0]);
  N = mxGetN(prhs[0]);

  D = mxGetPr(prhs[1]);
  M = mxGetM(prhs[1]);

  R = mxGetPr(prhs[2]);
  T = mxGetPr(prhs[3]);

  P = mxGetPr(prhs[4]);
  PM =  mxGetM(prhs[4]);
  PN =  mxGetN(prhs[4]);

  /*allocate for output*/
  plhs[0] = mxCreateDoubleMatrix(M, N, mxREAL);
  L = mxGetPr(plhs[0]);

  T1 = T[0]; 
  T2 = T[1]; 
  T3 = T[2];
  R11 = R[0]; 
  R12 = R[3]; 
  R13 = R[6]; 
  R21 = R[1]; 
  R22 = R[4]; 
  R23 = R[7]; 
  R31 = R[2]; 
  R32 = R[5]; 
  R33 = R[8];

  w = (double)PM/2;
  h = (double)PN/2;
  wfov = w/fov;

  /*
  mexPrintf("M=%d\n",M);
  mexPrintf("N=%d\n",N);
  mexPrintf("w=%f\n",w);
  mexPrintf("h=%f\n",h);
  */

  for (ii=0; ii<N; ii++){    
    Y1 = Y[0 + ii*3];
    Y2 = Y[1 + ii*3];
    Y3 = Y[2 + ii*3];

    for (jj=0; jj<M; jj++){
      D1 = D[jj + ii*M];
	/* mexPrintf("%f\n",D1); */
      if (D1 > 0){

	/*	mexPrintf("%f %f %f %f\n",Y1,Y2,Y3,D1);
		mexPrintf("%f %f %f\n",T1,T2,T3); */

	Z1 = D1*Y1 - T1;
	Z2 = D1*Y2 - T2;
	Z3 = D1*Y3 - T3;

	b1 = R11*Z1 + R12*Z2 + R13*Z3;
	b2 = R21*Z1 + R22*Z2 + R23*Z3;
	b3 = R31*Z1 + R32*Z2 + R33*Z3;

	/* mexPrintf("%f %f %f\n",b1,b2,b3); */

	a = sqrt(b1*b1 + b2*b2 + b3*b3);

	Z1 = b1/a;
	Z2 = b2/a;
	Z3 = b3/a;

	th = atan2(Z1,Z3);
	phi = asin(Z2);

	u = th*wfov + w;
	v = phi*wfov + h;

	ui = round(u);
	vi = round(v);
	if (ui==PM)
	  ui = PM-1;
	if (vi==PN)
	  vi = PN-1;

	/* mexPrintf("%d %d %d %d \n",ii,jj,ui,vi); */

	prep = (int)P[ui + vi*PM];

	/* mexPrintf("%f %f %f\n",Z1,Z2,Z3); 
	   mexPrintf("%d %d %d %d\n",jj,prep,ui,vi); */

	if (prep==(jj+1))
	  if (D[jj + ii*M]>0)
	    L[jj + ii*M] = D[jj + ii*M];
	  else
	    L[jj + ii*M] = NAN;
	else
	  L[jj + ii*M] = NAN;
      }
      else
	L[jj + ii*M] = NAN;
    }
  }
}

