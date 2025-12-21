/*
iminterp mex
(c) Michal Havlena, Mar 2008
    Revised (bicubic) Jan 2011, inspired by code of Stephane Zaleski

bicubic / bilinear / NN image interpolation (.mex)
imout = iminterp(imin,pts,mode)
imin ... source image
pts ... positions to be interpolated (struct) [x,y]
        floating point x and y coordinates
        size of this structure governs the size of output
imout ... interpolated output image
mode ... 2 = bicubic, 1 = bilinear (default), 0 = nearest neighbour
*/

#include "mex.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define PUTPIXEL(img, x, y, z, val) *(img + ((z)*oim_size + (x)*oheight + (y))) = (val)
#define GETPIXEL(img, x, y, z) *(img + ((z)*im_size + (x)*height + (y)))


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  mxArray *tmp;
  unsigned char *imin, *imout;
  double *xpts, *ypts;
  int *dim, *dim2;
  int odim[3];
  int width, height, im_size;
  int owidth, oheight, oim_size;
  int i, j;
  unsigned char ival;
  double u, v, f, g, ival1, ival2;
  int w11, w12, w21, w22;
  char mode;
  int k, l;
  int w10, w13, w20, w23;
  static int Ainv[16][16] = {
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
    {-3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0},
    {2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0},
    {0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0},
    {-3,0,3,0,0,0,0,0,-2,0,-1,0,0,0,0,0},
    {0,0,0,0,-3,0,3,0,0,0,0,0,-2,0,-1,0},
    {9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1},
    {-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1},
    {2,0,-2,0,0,0,0,0,1,0,1,0,0,0,0,0},
    {0,0,0,0,2,0,-2,0,0,0,0,0,1,0,1,0},
    {-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1},
    {4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1}
  };
  double xv[16], av[16];

  /* Check for proper number of arguments. */
  if ((nrhs > 3)||(nrhs < 2)) {
    mexErrMsgTxt("Wrong number of input arguments!");
  } else if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments!");
  } else if (!mxIsStruct(prhs[1])) {
    mexErrMsgTxt("The second input argument must be a structure!");
  }

  /* LOAD input */
  imin = (unsigned char *)mxGetPr(prhs[0]);
  dim = (int*)mxGetDimensions(prhs[0]);
  width = dim[1];
  height = dim[0];
  im_size = width*height;

  xpts = (double *)mxGetPr(mxGetField(prhs[1],0,"x"));
  dim2 = (int*)mxGetDimensions(mxGetField(prhs[1],0,"x"));
  owidth = dim2[1];
  oheight = dim2[0];
  oim_size = owidth*oheight;

  ypts = (double *)mxGetPr(mxGetField(prhs[1],0,"y"));
  dim2 = (int*)mxGetDimensions(mxGetField(prhs[1],0,"y"));
  if ((dim2[0] != oheight)||(dim2[1] != owidth)) {
    mexErrMsgTxt("Different matrix dimensions!");
  }

  if (nrhs > 2) {
    mode = mxGetScalar(prhs[2]);
  } else mode = 1;

  /* Create matrix for the return argument. */
  odim[0] = dim2[0];
  odim[1] = dim2[1];
  odim[2] = dim[2];
  plhs[0] = mxCreateNumericArray(3,odim,mxUINT8_CLASS,mxREAL);
  imout = (unsigned char *)mxGetPr(plhs[0]);

  for(i = 0; i < owidth; i++) {
    for(j = 0; j < oheight; j++) {
      u = *(xpts++)-1;
      v = *(ypts++)-1;
      w11 = (int)floor(u);
      w12 = (int)ceil(u);
      w21 = (int)floor(v);
      w22 = (int)ceil(v);
      f = u - w11;
      g = v - w21;

      if (w11 >= 0 && w12 < width && w21 >= 0 && w22 < height) {
        switch (mode) {
          case 2:
            w10 = w10 > 0 ? w11-1 : w12;
            w13 = w12 < width-1 ? w12+1 : w11;
            w20 = w20 > 0 ? w21-1 : w22;
            w23 = w22 < height-1 ? w22+1 : w21;
            xv[0]  = (double)GETPIXEL(imin, w11, w21, 0);
            xv[1]  = (double)GETPIXEL(imin, w12, w21, 0);
            xv[2]  = (double)GETPIXEL(imin, w11, w22, 0);
            xv[3]  = (double)GETPIXEL(imin, w12, w22, 0);
            xv[4]  = 0.5*(xv[1] - (double)GETPIXEL(imin, w10, w21, 0));
            xv[5]  = 0.5*((double)GETPIXEL(imin, w13, w21, 0) - xv[0]);
            xv[6]  = 0.5*(xv[3] - (double)GETPIXEL(imin, w10, w22, 0));
            xv[7]  = 0.5*((double)GETPIXEL(imin, w13, w22, 0) - xv[2]);
            xv[8]  = 0.5*(xv[2] - (double)GETPIXEL(imin, w11, w20, 0));
            xv[9]  = 0.5*(xv[3] - (double)GETPIXEL(imin, w12, w20, 0));
            xv[10] = 0.5*((double)GETPIXEL(imin, w11, w23, 0) - xv[0]);
            xv[11] = 0.5*((double)GETPIXEL(imin, w12, w23, 0) - xv[1]);
            xv[12] = 0.5*(xv[6] + xv[9] + 0.5*((double)GETPIXEL(imin, w10, w20, 0) - xv[3]));
            xv[13] = 0.5*(xv[7] - xv[8] - 0.5*((double)GETPIXEL(imin, w13, w20, 0) - xv[2]));
            xv[14] = 0.5*(xv[11] - xv[4] - 0.5*((double)GETPIXEL(imin, w10, w23, 0) - xv[1]));
            xv[15] = 0.5*(-xv[5] - xv[10] + 0.5*((double)GETPIXEL(imin, w13, w23, 0) - xv[0]));
            for (k = 0; k < 16; k++) {
              av[k] = 0.0;
              for (l = 0; l < 16; l++) av[k] += Ainv[k][l]*xv[l];
            }
            ival1 = 0.0;
            for (k = 3; k >= 0; k--) ival1 = ival1*g + ((av[4*k+3]*f + av[4*k+2])*f + av[4*k+1])*f + av[4*k];
            ival = (int)(ival1 < 0 ? 0 : (ival1 > 255 ? 255 : ival1));
            break;
          case 1:
            ival1 = (1-f)*(double)GETPIXEL(imin, w11, w21, 0) + f*(double)GETPIXEL(imin, w12, w21, 0);
            ival2 = (1-f)*(double)GETPIXEL(imin, w11, w22, 0) + f*(double)GETPIXEL(imin, w12, w22, 0);
            ival = (int)((1-g)*ival1 + g*ival2);
            break;
          case 0:
            ival = GETPIXEL(imin, (int)u, (int)v, 0);
            break;
        }
        PUTPIXEL(imout, i, j, 0, ival);
        switch (mode) {
          case 2:
            w10 = w10 > 0 ? w11-1 : w12;
            w13 = w12 < width-1 ? w12+1 : w11;
            w20 = w20 > 0 ? w21-1 : w22;
            w23 = w22 < height-1 ? w22+1 : w21;
            xv[0]  = (double)GETPIXEL(imin, w11, w21, 1);
            xv[1]  = (double)GETPIXEL(imin, w12, w21, 1);
            xv[2]  = (double)GETPIXEL(imin, w11, w22, 1);
            xv[3]  = (double)GETPIXEL(imin, w12, w22, 1);
            xv[4]  = 0.5*(xv[1] - (double)GETPIXEL(imin, w10, w21, 1));
            xv[5]  = 0.5*((double)GETPIXEL(imin, w13, w21, 1) - xv[0]);
            xv[6]  = 0.5*(xv[3] - (double)GETPIXEL(imin, w10, w22, 1));
            xv[7]  = 0.5*((double)GETPIXEL(imin, w13, w22, 1) - xv[2]);
            xv[8]  = 0.5*(xv[2] - (double)GETPIXEL(imin, w11, w20, 1));
            xv[9]  = 0.5*(xv[3] - (double)GETPIXEL(imin, w12, w20, 1));
            xv[10] = 0.5*((double)GETPIXEL(imin, w11, w23, 1) - xv[0]);
            xv[11] = 0.5*((double)GETPIXEL(imin, w12, w23, 1) - xv[1]);
            xv[12] = 0.5*(xv[6] + xv[9] + 0.5*((double)GETPIXEL(imin, w10, w20, 1) - xv[3]));
            xv[13] = 0.5*(xv[7] - xv[8] - 0.5*((double)GETPIXEL(imin, w13, w20, 1) - xv[2]));
            xv[14] = 0.5*(xv[11] - xv[4] - 0.5*((double)GETPIXEL(imin, w10, w23, 1) - xv[1]));
            xv[15] = 0.5*(-xv[5] - xv[10] + 0.5*((double)GETPIXEL(imin, w13, w23, 1) - xv[0]));
            for (k = 0; k < 16; k++) {
              av[k] = 0.0;
              for (l = 0; l < 16; l++) av[k] += Ainv[k][l]*xv[l];
            }
            ival1 = 0.0;
            for (k = 3; k >= 0; k--) ival1 = ival1*g + ((av[4*k+3]*f + av[4*k+2])*f + av[4*k+1])*f + av[4*k];
            ival = (int)(ival1 < 0 ? 0 : (ival1 > 255 ? 255 : ival1));
            break;
          case 1:
            ival1 = (1-f)*(double)GETPIXEL(imin, w11, w21, 1) + f*(double)GETPIXEL(imin, w12, w21, 1);
            ival2 = (1-f)*(double)GETPIXEL(imin, w11, w22, 1) + f*(double)GETPIXEL(imin, w12, w22, 1);
            ival = (int)((1-g)*ival1 + g*ival2);
            break;
          case 0:
            ival = GETPIXEL(imin, (int)u, (int)v, 1);
            break;
        }
        PUTPIXEL(imout, i, j, 1, ival);
        switch (mode) {
          case 2:
            w10 = w10 > 0 ? w11-1 : w12;
            w13 = w12 < width-1 ? w12+1 : w11;
            w20 = w20 > 0 ? w21-1 : w22;
            w23 = w22 < height-1 ? w22+1 : w21;
            xv[0]  = (double)GETPIXEL(imin, w11, w21, 2);
            xv[1]  = (double)GETPIXEL(imin, w12, w21, 2);
            xv[2]  = (double)GETPIXEL(imin, w11, w22, 2);
            xv[3]  = (double)GETPIXEL(imin, w12, w22, 2);
            xv[4]  = 0.5*(xv[1] - (double)GETPIXEL(imin, w10, w21, 2));
            xv[5]  = 0.5*((double)GETPIXEL(imin, w13, w21, 2) - xv[0]);
            xv[6]  = 0.5*(xv[3] - (double)GETPIXEL(imin, w10, w22, 2));
            xv[7]  = 0.5*((double)GETPIXEL(imin, w13, w22, 2) - xv[2]);
            xv[8]  = 0.5*(xv[2] - (double)GETPIXEL(imin, w11, w20, 2));
            xv[9]  = 0.5*(xv[3] - (double)GETPIXEL(imin, w12, w20, 2));
            xv[10] = 0.5*((double)GETPIXEL(imin, w11, w23, 2) - xv[0]);
            xv[11] = 0.5*((double)GETPIXEL(imin, w12, w23, 2) - xv[1]);
            xv[12] = 0.5*(xv[6] + xv[9] + 0.5*((double)GETPIXEL(imin, w10, w20, 2) - xv[3]));
            xv[13] = 0.5*(xv[7] - xv[8] - 0.5*((double)GETPIXEL(imin, w13, w20, 2) - xv[2]));
            xv[14] = 0.5*(xv[11] - xv[4] - 0.5*((double)GETPIXEL(imin, w10, w23, 2) - xv[1]));
            xv[15] = 0.5*(-xv[5] - xv[10] + 0.5*((double)GETPIXEL(imin, w13, w23, 2) - xv[0]));
            for (k = 0; k < 16; k++) {
              av[k] = 0.0;
              for (l = 0; l < 16; l++) av[k] += Ainv[k][l]*xv[l];
            }
            ival1 = 0.0;
            for (k = 3; k >= 0; k--) ival1 = ival1*g + ((av[4*k+3]*f + av[4*k+2])*f + av[4*k+1])*f + av[4*k];
            ival = (int)(ival1 < 0 ? 0 : (ival1 > 255 ? 255 : ival1));
            break;
          case 1:
            ival1 = (1-f)*(double)GETPIXEL(imin, w11, w21, 2) + f*(double)GETPIXEL(imin, w12, w21, 2);
            ival2 = (1-f)*(double)GETPIXEL(imin, w11, w22, 2) + f*(double)GETPIXEL(imin, w12, w22, 2);
            ival = (int)((1-g)*ival1 + g*ival2);
            break;
          case 0:
            ival = GETPIXEL(imin, (int)u, (int)v, 2);
            break;
        }
        PUTPIXEL(imout, i, j, 2, ival);
      } else {
        PUTPIXEL(imout, i, j, 0, 0);
        PUTPIXEL(imout, i, j, 1, 0);
        PUTPIXEL(imout, i, j, 2, 0);
      }
    }
  }
  return;
}
