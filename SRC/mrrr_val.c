/* Parallel computation of eigenvalues and symmetric tridiagonal 
 * matrix T, given by its diagonal elements D and its super-/sub-
 * diagonal elements E.
 *
 * Copyright (c) 2010, RWTH Aachen University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or 
 * without modification, are permitted provided that the following
 * conditions are met:
 *   * Redistributions of source code must retain the above 
 *     copyright notice, this list of conditions and the following
 *     disclaimer.
 *   * Redistributions in binary form must reproduce the above 
 *     copyright notice, this list of conditions and the following 
 *     disclaimer in the documentation and/or other materials 
 *     provided with the distribution.
 *   * Neither the name of the RWTH Aachen University nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL RWTH 
 * AACHEN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE.
 *
 * Coded by Matthias Petschow (petschow@aices.rwth-aachen.de),
 * September 2010, Version 1.2
 *
 * This code was the result of a collaboration between 
 * Matthias Petschow and Paolo Bientinesi. When you use this 
 * code, kindly reference the paper:
 *
 * "MR3-SMP: A Symmetric Tridiagonal Eigensolver for Multi-Core 
 * Architectures" by Matthias Petschow and Paolo Bientinesi, 
 * RWTH Aachen, Technical Report AICES-2010/10-2 (submitted to 
 * Parallel Computing).
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <pthread.h>
#include "global.h"
#include "mrrr.h"
#include "structs.h"


static inline void clean_up(double*, double*, double*, int*);

static inline 
int find_eigval_irreducible(int max_nthreads, int n, int il, int iu, 
			    double *restrict D, double *restrict E, double *restrict E2, 
			    double *restrict W, double *restrict Werr,  
			    int *restrict Windex, double *restrict gersch, 
			    tol_t *tolstruct, double *work, int *iwork);

static inline void *eigval_approx(void*);

static inline aux1_t *create_auxarg1(int tid, int il, int iu, int my_il, int my_iu, 
				     int n, double *D, double *E, double *E2, double *W, 
				     double *Werr, int *Windex, double *gersch, tol_t* tolstruct); 
static inline void retrieve_auxarg1(aux1_t *arg, int *tid, int *il, 
				    int *iu, int *my_il, int *my_iu, int *n, double **D, 
				    double **E, double **E2, double **W,  
				    double **Werr, int **Windex, double **gersch, tol_t** tolstruct);
static inline int compute_root_rrr(bool, bool, int, int, int, 
				   int, int, int, double, double, 
				   in_t*, val_t*, tol_t*, double*, 
				   int*, int*, double*, double*);
static inline int refine_eigval_approx(int, int, int, int, int, int, 
				       int, double, double, in_t*, 
				       val_t*, tol_t*, double*, int*);
static void *eigval_refine(void*);
static inline aux2_t *create_auxarg2(int, int, int, int, double, double*, 
				     double*, val_t*, tol_t*);
static inline void retrieve_auxarg2(aux2_t*, int*, int*, int*, int*, 
				    double*, double**, double**, val_t**, 
				    tol_t**);


#define HUNDRED      100.0
#define FOURTH         0.25
#define FRACTION_DQDS  0.5
#define FUDGE          2.0
#define RAND_FACTOR    8.0


/* Computes all or a subset of eigenvalues of a symmetric tridiagonal 
 * matrix.
 *
 * Note: this routine is, with very little changes, LAPACK's 'dlarre'
 * with the bisection routines replaced by multi-threaded versions; 
 * it can most likely be optimized in terms of speed, memory 
 * requirement and the clarity of the code.
 */
int mrrr_val(char *jobz, char *range, int nthreads, in_t *Dstruct,
	     val_t *Wstruct, tol_t *tolstruct)
{
  /* Inputs */
  int    n                = Dstruct->n;
  double *restrict D      = Dstruct->D;
  double *restrict E      = Dstruct->E;
  int    *restrict isplit = Dstruct->isplit;

  int    *mp              = Wstruct->mp;
  double *vlp             = Wstruct->vlp;
  double *vup             = Wstruct->vup;
  int    *ilp             = Wstruct->ilp;
  int    *iup             = Wstruct->iup;
  double *restrict W      = Wstruct->W;
  double *restrict Werr   = Wstruct->Werr;
  double *restrict Wgap   = Wstruct->Wgap;
  int    *restrict iblock = Wstruct->iblock;
  int    *restrict Windex = Wstruct->Windex;
  double *restrict gersch = Wstruct->gersch;

  int    max_nthreads    = nthreads;
  bool   force_bisection = FORCE_BISECTION;

  enum range_enum {allrng=1, valrng=2, indrng=3} irange;
  
  /* Others */
  int    IZERO=0, IONE=1, ITWO=2, ITHREE=3;
  double DZERO = 0.0;
  int    i, info;
  double gl, gu, eold, emax, eabs, spdiam;
  double *E2, *work;
  int    *iwork;
  double intervals[2];
  int    negcounts[2];
  int    nvals;
  bool   useDQDS;
  int    idummy;
  double dummy;
  int    ibgn, iend, nbl;
  int    wbgn, wend, mbl;
  double rtol;
  int    jbl, j;
  double tmp1;
  int    ilow, iupp;
  double sigma;
  int    sgndef;
  bool   allrng_no_bisec;

  int itmax;
  double tnorm, atoli;
  double wl, wlu, wu, wul, wkill;
  int nwl, nwu;
  int idiscl, idiscu;
  int im, iw, je, jee, jdisc;
  double *restrict Wcopy;

  /* Decode range */
  if (range[0] == 'A' || range[0] == 'a') {
    irange = allrng;
  } else if (range[0] == 'V' || range[0] == 'v') {
    irange = valrng;
  } else if (range[0] == 'I' || range[0] == 'i') {
    irange = indrng;
  } else {
    return(1);
  }
  if (irange == indrng && *ilp == 1 && *iup == n)
    irange = allrng;

  /* Allocate work space */
  E2    = (double *) malloc(   n * sizeof(double) );
  assert(E2 != NULL);

  Wcopy = (double *) malloc(   n * sizeof(double) );
  assert(Wcopy != NULL);

  Dstruct->E2 = E2;

  work  = (double *) malloc( 4*n * sizeof(double) );
  assert (work != NULL);
  iwork = (int *)    malloc( 3*n * sizeof(int) );
  assert (iwork != NULL);

  /* Treat n=1 case. Note: should not happen since blocked before 
  * by calling the sequential code for small cases */
  if (n == 1) {
    if ( (irange == allrng) || 
         (irange == valrng && D[0] > *vlp && D[0] <= *vup) || 
	 (irange == indrng && *ilp == 1   && *iup == 1) ) {
      *mp        = 1;
      W[0]       = D[0];
      Werr[0]    = 0.0;
      Wgap[0]    = 0.0;
      iblock[0]  = 1;
      Windex[0]  = 1;
      gersch[0]  = D[0];
      gersch[1]  = D[0];
    }
    E[0] = 0.0;
    return(0);
  }

  /* Initialize E2 */
  for (i=0; i<n; i++) E2[i] = E[i]*E[i];

  /* Compute geschgorin disks and spectral diameter */
  gl     = D[0];
  gu     = D[0];
  eold   = 0.0;
  emax   = 0.0;
  E[n-1] = 0.0;

  for (i=0; i<n; i++) {
    eabs = fabs(E[i]);
    if (eabs >= emax) emax = eabs;
    tmp1 = eabs + eold;
    gersch[2*i] = D[i] - tmp1;
    gl = fmin(gl, gersch[2*i]);
    gersch[2*i+1] = D[i] + tmp1;
    gu = fmax(gu, gersch[2*i+1]);
    eold = eabs;
  }
  /* Min. pivot allowed in the Sturm sequence of T */
  tolstruct->pivmin = DBL_MIN * fmax(1.0, emax*emax);
  /* Estimate of spectral diameter */
  Dstruct->spdiam = gu - gl;

  /* Fudge Gershgorin bounds */
  tnorm = fmax(fabs(gl),fabs(gu));
  gl = gl - 2*tnorm*DBL_EPSILON*n - 4*tolstruct->pivmin;
  gu = gu + 2*tnorm*DBL_EPSILON*n + 4*tolstruct->pivmin;

  /* Check if all values are contained in (vl,vu] */
  if (irange == valrng && *vlp < gl && *vup >= gu)
    irange = allrng;

    /* compute splitting points */
  dlarra_(&n, D, E, E2, &tolstruct->split, &Dstruct->spdiam, 
	  &Dstruct->nsplit, isplit, &info);
  assert(info == 0);

  /* Find number of eigenvalues to compute */
  if (irange == allrng) {
    *ilp = 1;
    *iup = n;
  } else if (irange == valrng) {
    intervals[0] = *vlp; intervals[1] = *vup;
    
    /* Find negcount at boundaries; needs work of dim(n) 
     * and iwork of dim(n) */
    dlaebz_(&IONE, &IZERO, &n, &IONE, &IONE, &IZERO, &DZERO, &DZERO, 
	    &tolstruct->pivmin, D, E, E2, &idummy, intervals, &dummy, 
	    &idummy, negcounts, work, iwork, &info);
    assert(info == 0);
    
    *ilp = negcounts[0] + 1;
    *iup = negcounts[1];
  }
  nvals = *iup - *ilp + 1;
  *mp = nvals;

  /* If jobz="C" all done */
  if (jobz[0] == 'C' || jobz[0] == 'c') {
    clean_up(E2, Wcopy, work, iwork);
    *mp = nvals;
    return(0);
  }

  /* Force use of bisection in case it is faster than dqds */
  max_nthreads = nthreads;
  if (max_nthreads > ceil(SWITCH_TO_BISEC*nvals / (double) n))
    force_bisection = true;

  /* Usualy use dqds if all eigenvalues are to be computed */
  useDQDS = (irange == allrng && !force_bisection) ? true : false;
  
  if (useDQDS) {    
    wl = gl;
    wu = gu;
    *vlp = gl;
    *vup = gu;
  } else if (!useDQDS) {

    /* Find interval that contains all desired eigenvalues */
    if (irange == indrng) {
      itmax = ( log( tnorm+tolstruct->pivmin )-log( tolstruct->pivmin ) ) /  
	log(2) + 2;
      atoli = 2*(2*DBL_MIN + 2*tolstruct->pivmin);
      work[n] = gl; work[n+1] = gl; work[n+2] = gu;
      work[n+3] = gu; work[n+4] = gl; work[n+5] = gu;
      iwork[0] = -1; iwork[1] = -1; iwork[2] = n + 1;
      iwork[3] = n + 1; iwork[4] = *ilp - 1; iwork[5] = *iup;
      
      dlaebz_(&ITHREE, &itmax, &n, &ITWO, &ITWO, &IZERO, &atoli, 
	      &tolstruct->RQtol, &tolstruct->pivmin, D, E, E2, &iwork[4], 
	      &work[n], &work[n+4], &idummy, iwork, W, iblock, &info);
      assert(info == 0);
      
      if (iwork[5] == *iup) {
	wl = work[n]; wlu = work[n+2];
	wu = work[n+3]; wul = work[n+1];
	nwl = iwork[0]; nwu = iwork[3];
      }else {
	wl = work[n+1]; wlu = work[n+3];
	wu = work[n+2]; wul = work[n];
	nwu = iwork[2]; nwl = iwork[1];
      } 
      *vlp = wl; 
      *vup = wu;
    } else if (irange == valrng) {
      wl = *vlp;
      wu = *vup;
    } else {
      wl = gl;
      wu = gu;
      *vlp = gl; 
      *vup = gu;
    }   
  }

  /* Loop over unreduced blocks. 
     NOTE: Eventually the code below should only produce tasks which can be 
     executed threads in a task pool (as in mrrr_vec). 
     In this case, (1) tasks can be made smaller to improve load balancing, and
     (2) the computation of different blocks is done in parallel */
  *mp  = 0;
  ibgn = 0;
  wbgn = 0;
  
  for (jbl=0; jbl<Dstruct->nsplit; jbl++) {
    
    iend = isplit[jbl] - 1;
    nbl  = iend - ibgn + 1;

    /* Deal with 1x1 block immediately */
    if (nbl == 1) {
      if ( (irange==allrng) ||
  	   (irange==valrng && D[ibgn]>wl && D[ibgn]<=wu) ||
  	   (irange==indrng && (D[ibgn]+tolstruct->pivmin)>wl 
	    && (D[ibgn]-tolstruct->pivmin)<=wu) ) {
	
  	W[*mp]      = D[ibgn];
  	Wcopy[*mp]  = D[ibgn];
  	Werr[*mp]   = 0.0;
  	Werr[*mp]   = 0.0;
  	iblock[*mp] = jbl+1;
  	Windex[*mp] = 1;
  	(*mp)++;
  	wbgn++;
      }
      E[iend] = 0.0;
      ibgn  = iend + 1;
      continue;
    }

    /* Find outer bounds GL, GU for block and spectral diameter */
    gl = D[ibgn];
    gu = D[ibgn];
    for (i=ibgn; i<=iend; i++) {
      gl = fmin(gl, gersch[2*i]  );
      gu = fmax(gu, gersch[2*i+1]);
    }
    spdiam = gu - gl;
    gl = gl - 2*tnorm*DBL_EPSILON*nbl - 4*tolstruct->pivmin;
    gu = gu + 2*tnorm*DBL_EPSILON*nbl + 4*tolstruct->pivmin;

    /* If not all eigenvalues wanted and bisection is not forced:
     * 1) count eigenvalues in block, skip block if no
     * 2) decide if use DQDS is more efficient */
    if (irange == allrng) {
      ilow = 1;
      iupp = nbl;
      mbl  = nbl;
      wend = wbgn + mbl - 1;
    } else { // if (!useDQDS) {
      intervals[0] = fmax(wl, gl); intervals[1] = fmin(wu, gu);
      
      /* Find negcount at boundaries; needs work of dim(n)
       * and iwork of dim(n) */
      dlaebz_(&IONE, &IZERO, &nbl, &IONE, &IONE, &IZERO, &DZERO, &DZERO,
  	      &tolstruct->pivmin, &D[ibgn], &E[ibgn], &E2[ibgn], &idummy, 
	      intervals, &dummy, &idummy, negcounts, work, iwork, &info);
      assert(info == 0);
	
      ilow = negcounts[0] + 1;
      iupp = negcounts[1];

    }
    mbl = iupp - ilow + 1;
    *mp  += mbl;
    
    if (mbl == 0) {
      E[iend] = 0.0;
      ibgn = iend + 1;
      continue;
    } else {

      /* Use DQDS if many eigenvalues wanted */
      if (mbl > FRACTION_DQDS*nbl && !force_bisection)
  	useDQDS = true;
      else
  	useDQDS = false;

      wend = wbgn + mbl - 1;

      // should use memset here
      for (i=wbgn; i<=wend; i++)
  	iblock[i] = jbl+1;
    
      /* Find crude approximation of the eigenvalues ilow:iupp */
      info = find_eigval_irreducible(max_nthreads, nbl, ilow, iupp,
  				     &D[ibgn], &E[ibgn], &E2[ibgn],
  				     &W[wbgn], &Werr[wbgn], &Windex[wbgn],
  				     &gersch[2*ibgn], tolstruct, work, iwork);
      assert(info == 0);

      // should use memcopy here
      for (i=wbgn; i<=wend; i++)
  	Wcopy[i] = W[i];
      
      /* Calculate gaps */
      for (i=wbgn; i<wend; i++) {
  	Wgap[i] = fmax(0.0,(W[i+1] - Werr[i+1]) - (W[i] + Werr[i]));
      }
      Wgap[wend] = fmax(0.0, wu - (W[wend] + Werr[wend]));

    } /* end if mbl==0 */

    /* Compute the root representation of the block and place it
     * in D[ibgn:iend] and E[ibgn:iend], returning sgndef=+/-1,
     * the shift sigma, and a possibly refined spectral diameter */
    allrng_no_bisec = (irange == allrng && !force_bisection);

    info =  compute_root_rrr(allrng_no_bisec, useDQDS,
  			     ibgn, iend, wbgn, wend, ilow,
  			     iupp, gl, gu, Dstruct, Wstruct,
  			     tolstruct, work, iwork, &sgndef,
  			     &sigma, &spdiam);
    assert(info == 0);

    /* Compute desired eigenvalues by bisection or dqds */
    if (!useDQDS) {
      /* Bisection has been used before, now refine */
      
      info = refine_eigval_approx(max_nthreads, ibgn, iend, wbgn,
  				  wend, ilow, iupp, sigma, spdiam,
  				  Dstruct, Wstruct, tolstruct, work,
  				  iwork);
      assert(info == 0);

    } else {
      /* use DQDS */
      
      /* qd = work[0:4*n-1] */
      rtol = 4.0 * log((double) nbl) * DBL_EPSILON;
      
      j = ibgn;
      for (i=0; i<nbl-1; i++) {
  	work[ 2*i ] = fabs(D[j]);
  	work[2*i+1] = E[j]*E[j]*work[2*i];
  	j++;
      }
      work[2*(nbl-1)]   = fabs(D[iend]);
      work[2*(nbl-1)+1] = 0.0;

      /* Compute all eigenvalues with DQDS */
      dlasq2_(&nbl, work, &info);
      assert(info == 0);

      if (sgndef > 0) {
  	je = wbgn;
  	for (i=ilow; i<=iupp; i++) {
  	  /* sort to increasing order */
  	  W[je] = work[nbl-i];
  	  iblock[je] = jbl+1;
  	  Windex[je] = i;
  	  je++;
  	}
      } else {
  	je = wbgn;
  	for (i=ilow; i<=iupp; i++) {
  	  W[je]      = -work[i-1];
  	  iblock[je] = jbl+1;
  	  Windex[je] = i;
  	  je++;
  	}
      }

      for (i=wbgn; i<=wend; i++) {
  	Werr[i] = rtol * fabs( W[i] );
      }
      
      for (i=wbgn; i<wend; i++) {
  	Wgap[i] = fmax(0.0, (W[i+1]-Werr[i+1]) - (W[i]+Werr[i]));
      }
      Wgap[wend] = fmax(0.0, (*vup-sigma) - (W[wend]+Werr[wend]) );

    } /* end bisection or dqds */
      
    ibgn = iend + 1;
    wbgn = wend + 1;
  } /* end of loop over unreduced blocks */


  /* Remove eigenvalues if too many are computed. 
     NOTE: it would be better (i.e., cheaper) to run twice over the loop 
     of unreduced blocks and first compute approximations to lambda(T), 
     discard if too many, and second only refine the
     ones we actually keep */
  if (*mp > nvals) {
    
    idiscl = *ilp - 1 - nwl;
    idiscu = nwu - *iup;
    
    if(idiscl > 0) {
      im = 0;
      for (je =0; je<*mp; je++) {
  	/* Remove some of the smallest eigenvalues from the left so that */
  	/* at the end idiscl =0. Move all eigenvalues up to the left. */
  	if(Wcopy[je] < wlu && idiscl > 0) {
  	  idiscl--;
  	} else {
  	  W[im] = W[je];
    	  Wcopy[im] = Wcopy[je];
  	  Werr[im] = Werr[je];
  	  Wgap[im] = Wgap[je];
  	  Windex[im] = Windex[je];
  	  iblock[im] = iblock[je];
  	  im++;
  	}
      }
      *mp = im; /* number of eigenvalues once we removed from left */
    }

    if( idiscu > 0 ) {
      /* Remove some of the largest eigenvalues from the right so that */
      /*  at the end idiscu =0. Move all eigenvalues up to the left. */
      im = *mp;
      for (je = *mp-1; je >=0; je--) {
    	if (Wcopy[je] > wul && idiscu > 0) {
    	  idiscu--;
    	} else {
    	  im--;
    	  W[im] = W[je];
    	  Wcopy[im] = Wcopy[je];
    	  Werr[im] = Werr[je];
  	  Wgap[im] = Wgap[je];
    	  Windex[im] = Windex[je];
    	  iblock[im] = iblock[je];
    	}
      }
      /* Copy values from right to left */
      jee = 0;
      for (je = im; je<*mp; je++) {
  	W[jee] = W[je];
  	Wcopy[jee] = Wcopy[je];
  	Werr[jee] = Werr[je];
	Wgap[jee] = Wgap[je];
  	Windex[jee] = Windex[je];
  	iblock[jee] = iblock[je];
  	jee++;
      }
      *mp -= im;
    }

    /* If still too many eigenvalues kill some of them. 
       NOTE: DLARRD says that this should only happend when 
       there is bad arithmetic, but in fact
       in can happen otherwise. Example: T_zenios.dat */
    if(idiscl > 0 || idiscu > 0) {
      
      if(idiscl > 0) {
  	wkill = wu;
  	for (jdisc=0; jdisc<idiscl; jdisc++) {
  	  iw = -1;
  	  for (je=0; je<*mp; je++) {
  	    if(iblock[je] !=0 && (Wcopy[je] < wkill || iw==-1)) {
  	      iw = je;
  	      wkill = Wcopy[je];
  	    }
  	  }
  	  iblock[iw] = 0;
  	}
      }
      
      if (idiscu > 0) {
  	wkill = wl;
  	for (jdisc=0; jdisc<idiscu; jdisc++) {
  	  iw = -1;
  	  for (je=0; je<*mp; je++) {
  	    if(iblock[je] !=0 && (Wcopy[je] > wkill || iw==-1)) {
  	      iw = je;
  	      wkill = Wcopy[je];
  	    }
  	  }
  	  iblock[iw] = 0;
  	}
      }
      
      /* Now erase all eigenvalues with iblock set to zero */
      im = 0;
      for (je=0; je<*mp; je++) {
  	if (iblock[je] != 0) {
  	  W[im] = W[je];
  	  Wcopy[im] = Wcopy[je];
  	  Werr[im] = Werr[je];
  	  Wgap[im] = Wgap[je];
  	  Windex[im] = Windex[je];
  	  iblock[im] = iblock[je];
  	  im++;
  	}
      }
      *mp = im;
    } /* end kill eigenvalues */
    
  } /* end removing eigenvalues */

  /* Save the number of computed eigenvalues */ 
  Wstruct->m = *mp;

  if (*mp < nvals) {
    /* Call another routine? */
    clean_up(E2, Wcopy, work, iwork);
    return(3);
  }

  clean_up(E2, Wcopy, work, iwork);
  return(0);
}




/* Free's on allocated memory of routine */
static inline   
void clean_up(double *E2, double *Wcopy, double *work, int *iwork)
{
  free(E2);
  free(Wcopy);
  free(work);
  free(iwork);
}



static inline 
aux1_t *create_auxarg1(int tid, int il, int iu, int my_il, int my_iu, int n,
		       double *D, double *E, double *E2, double *W, 
		       double *Werr, int *Windex, double *gersch, tol_t *tolstruct)
{
  aux1_t *arg;
  
  arg = (aux1_t *) malloc(sizeof(aux1_t));
  assert(arg != NULL);

  arg->tid = tid;
  arg->il = il;
  arg->iu = iu;
  arg->my_il = my_il;
  arg->my_iu = my_iu;
  arg->n = n;
  arg->D = D;
  arg->E = E;
  arg->E2 = E2;
  arg->W = W;
  arg->Werr = Werr;
  arg->Windex = Windex;
  arg->gersch = gersch;
  arg->tolstruct = tolstruct;

  return(arg);
}



static inline 
void retrieve_auxarg1(aux1_t *arg, int *tid, int *il, 
		      int *iu, int *my_il, int *my_iu, int *n, double **D, double **E, 
		      double **E2, double **W,  double **Werr, int **Windex, 
		      double **gersch, tol_t **tolstruct)
{
  *tid       = arg->tid;
  *il     = arg->il;
  *iu     = arg->iu;
  *my_il     = arg->my_il;
  *my_iu     = arg->my_iu;
  *n     = arg->n;
  *D     = arg->D;
  *E     = arg->E;
  *E2     = arg->E2;
  *W     = arg->W;
  *Werr     = arg->Werr;
  *Windex     = arg->Windex;
  *gersch     = arg->gersch;
  *tolstruct = arg->tolstruct;
  free(arg);
}




/* Compute the root representation of a block, placing it into
 * D[ibgn:iend] and E[ibgn:iend]; requires work of size 3*n;
 * Note: this part of the code is just a copy&paste of a part of
 * code initially residing where the function call is now */
static inline 
int compute_root_rrr(bool allrng_no_bisec, bool useDQDS,
		     int ibgn, int iend, int wbgn, int wend, int ilow, 
		     int iupp, double gl, double gu, in_t *Dstruct, 
		     val_t *Wstruct, tol_t *tolstruct, double *work, 
		     int *iwork, int *sgndefp, double *sigmap,
		     double *spdiamp)
{
  /* Inputs */
  int              nbl  = iend - ibgn + 1;
  int              mbl  = wend - wbgn + 1;
  int              n    = Dstruct->n;
  double *restrict D    = Dstruct->D;
  double *restrict E    = Dstruct->E;
  double *restrict E2   = Dstruct->E2;
  double           *vlp = Wstruct->vlp;
  double           *vup = Wstruct->vup;
  double *restrict W    = Wstruct->W;
  double *restrict Werr = Wstruct->Werr;
  double *restrict Wgap = Wstruct->Wgap;
  double pivmin         = tolstruct->pivmin;
  double rtl            = sqrt(DBL_EPSILON);

  /* Others */
  int    i, j, jtry, info;
  int    IONE=1, ITWO=2;
  double tmp0, tmp1;
  double isleft, isright, spdiam;
  double s1, s2;
  int    cnt, negcnt_lft, negcnt_rgt;
  double sigma;
  int    sgndef;
  double clwdth, avgap, tau;
  bool   noREP;
  int    off_L, off_invD;
  double Dpivot, Dmax;
  int    n_randvec, iseed[4] = {1,1,1,1};

  /* Get aproximations to extremal eigenvalues */
  if (allrng_no_bisec || useDQDS) {
    /* Case of DQDS */
    
    /* Find approximation of extremal eigenvalues of the block;
     * tmp0 and tmp1 one hold the eigenvalue and error */
    dlarrk_(&nbl, &IONE, &gl, &gu, &D[ibgn], &E2[ibgn],
	    &pivmin, &rtl, &tmp0, &tmp1, &info);
    assert(info == 0);
    
    isleft = fmax(gl, (tmp0 - tmp1) 
		  - HUNDRED*DBL_EPSILON*fabs(tmp0-tmp1));
    
    dlarrk_(&nbl, &nbl, &gl, &gu, &D[ibgn], &E2[ibgn],
	    &pivmin, &rtl, &tmp0, &tmp1, &info);
    assert(info == 0);
    
    isright = fmin(gu, (tmp0 + tmp1) 
		   + HUNDRED*DBL_EPSILON*fabs(tmp0+tmp1));

    spdiam = isright - isleft; 

  } else {
    /* Case of bisection */
    
    isleft  = fmax(gl, (W[wbgn] - Werr[wbgn]) 
		   - HUNDRED * DBL_EPSILON 
		   * fabs(W[wbgn] - Werr[wbgn]));
    
    isright = fmin(gu, (W[wend] + Werr[wend])
		   + HUNDRED * DBL_EPSILON 
		   * fabs(W[wend] - Werr[wend]));

    spdiam = gu - gl;
  }

  
  /* Decide where to shift for initial representation
   * 1) find quarter points in aproximate interval
   * 2) compute negcounts for those points
   * 3) select shifting to more crowded end of spectrum
   */
  
  /* Find quarter points s1 and s2 */ 
  if (allrng_no_bisec) {
    s1 = isleft  + FOURTH * spdiam;
    s2 = isright - FOURTH * spdiam;
  } else {
    if (useDQDS) {
      s1 = isleft  + FOURTH * spdiam;
      s2 = isright - FOURTH * spdiam;
    } else {
      tmp0 = fmin(isright, *vup) - fmax(isleft, *vlp);
      s1   = fmax(isleft,  *vlp) + FOURTH * tmp0;
      s2   = fmin(isright, *vup) - FOURTH * tmp0;
    }
  }

  /* Compute negcount at quarter points */
  if (mbl > 1) {
    dlarrc_("T", &nbl, &s1, &s2, &D[ibgn], &E[ibgn], &pivmin, 
	    &cnt, &negcnt_lft, &negcnt_rgt, &info);
    assert(info == 0);
  }
  
  /* Select shift for initial RRR of the block */
  if (mbl == 1) {
    sigma  = gl;
    sgndef = 1;
  } else if ( (negcnt_lft - ilow) >= (iupp - negcnt_rgt) ) {
    /* More eigenvalues to compute left of s1 than right of s2
     * so shift to left end of the spectrum */
    if (allrng_no_bisec) {
      sigma = fmax(isleft, gl);
    } else if (useDQDS) {
      sigma = isleft;
    } else {
      sigma = fmax(isleft, *vlp);
    }
    sgndef = 1;
  } else {
    if (allrng_no_bisec) {
      sigma = fmin(isright, gu);
    } else if (useDQDS) {
      sigma = isright;
    } else {
      sigma = fmin(isright, *vup);
    }
    sgndef = -1;
  }
  
  /* Define increment to perturb initial shift to find RRR
   * with not too much element growth */
  if (useDQDS) {   
    tau = spdiam * DBL_EPSILON * nbl + 2.0 * pivmin;  
    tau = fmax(tau, 2*DBL_EPSILON*fabs(sigma));
  } else {
    if (mbl > 1) {
      clwdth = W[wend] + Werr[wend] - W[wbgn] - Werr[wbgn];
      avgap  = fabs(clwdth / (double) (wend - wbgn) );
      if (sgndef > 0) {
	tau = fmax(Wgap[wbgn], avgap);
	tau *= 0.5;
	tau = fmax(tau, Werr[wbgn]);
      } else {
	tau = fmax(Wgap[wend-1], avgap);
	tau *= 0.5;
	tau = fmax(tau, Werr[wend]);
      }
    } else {
      tau = Werr[wbgn];
    }
  }
  
  /* Try to find initial RRR of block:
   * need work space of 3*n here to store D, L, D^-1 of possible
   * representation:
   * D_try      = work[0  :  n-1] 
   * L_try      = work[n  :2*n-1]
   * inv(D_try) = work[2*n:3*n-1] */

  off_L    = n;
  off_invD = 2*n;
  
  for (jtry = 0; jtry < MAX_TRY_ROOT; jtry++) {
    
    Dpivot  = D[ibgn] - sigma;
    work[0] = Dpivot;
    Dmax    = fabs(work[0]);
    j = ibgn;
    
    for (i = 0; i <nbl-1; i++) {
      work[i+off_invD] = 1.0 / work[i];
      tmp0 = E[j] * work[i+off_invD];
      work[i+off_L] = tmp0;
      Dpivot = (D[j+1] - sigma) - tmp0*E[j];
      work[i+1] = Dpivot;
      Dmax = fmax(Dmax, fabs(Dpivot));
      j++;
    }
    
    noREP = (Dmax > MAX_GROWTH*spdiam) ? true : false;
    
    /* Check if found representation is definite as it should */
    if (useDQDS && !noREP) {
      for (i=0; i<nbl; i++) {
	tmp0 = sgndef*work[i];    /* work[0:nbl-1] = D_try */
	if (tmp0 < 0.0) noREP = true;
      }
    }
    
    if (noREP) {
      /* If all eigenvalues are desired shift is made definite to 
       * use DQDS so we should not end here */
      if (jtry == MAX_TRY_ROOT-2) {
	if (sgndef == 1) { /* FP cmp okay */
	  sigma = gl - FUDGE*spdiam*DBL_EPSILON*nbl - FUDGE*2.0*pivmin;
	} else {
	  sigma = gu + FUDGE*spdiam*DBL_EPSILON*nbl + FUDGE*2.0*pivmin;
	}
      } else if (jtry == MAX_TRY_ROOT-1) {
	/* No initial representation could be found */
	return(2);
      } else {
	sigma -= sgndef*tau;
	tau   *= 2.0;
	continue;
      }
    } else {
      break;
    }  
  } /* end trying to find initial RRR of block */
  
  /* Save initial RRR and corresponding shift */
  E[iend] = sigma;
  memcpy(&D[ibgn], &work[0],  nbl    * sizeof(double) );
  memcpy(&E[ibgn], &work[n], (nbl-1) * sizeof(double) );
  
  /* Perturb initial RRR by small rel. random amount */
  if(mbl > 1) {
    n_randvec = 2*nbl;
    dlarnv_(&ITWO, iseed, &n_randvec, work);
    for (i=0; i<nbl-1; i++) {
      D[i+ibgn] *= 1.0 + DBL_EPSILON*RAND_FACTOR*work[i];
      E[i+ibgn] *= 1.0 + DBL_EPSILON*RAND_FACTOR*work[i+nbl];
    }
    D[iend] *= 1.0 + DBL_EPSILON*RAND_FACTOR*work[2*nbl-1];
  }
  
  *sgndefp = sgndef;
  *sigmap  = sigma;
  *spdiamp = spdiam;

  return(0);
}




static inline 
int refine_eigval_approx(int max_nthreads, int ibgn, int iend, 
			 int wbgn, int wend, int ilow, int iupp, 
			 double sigma, double spdiam, in_t *Dstruct, 
			 val_t *Wstruct, tol_t *tolstruct, double *work, 
			 int *iwork)
{
  int              nbl    = iend - ibgn + 1;
  int              mbl    = wend - wbgn + 1;
  int              n      = Dstruct->n;
  double *restrict D      = Dstruct->D;
  double *restrict E      = Dstruct->E;
  //  int              *mp    = Wstruct->mp;
  double           *vup   = Wstruct->vup;
  double *restrict W      = Wstruct->W;
  double *restrict Werr   = Wstruct->Werr;
  double *restrict Wgap   = Wstruct->Wgap;
  // int        *restrict Windex = Wstruct->Windex;
  double rtol1            = tolstruct->rtol1;
  double rtol2            = tolstruct->rtol2;
  double pivmin           = tolstruct->pivmin;

  /* Others */
  int            i, info;
  int            off_DE2;
  int            nthreads;
  pthread_t      *threads;
  pthread_attr_t attr;
  void           *status;
  int            chunk;
  aux2_t         *arg;
  int            rf_begin, rf_end;
  int            offset;

  /* work  for sequential dlarrb = work[0:2*n-1]
   * iwork for sequential dlarrb = iwork[0:2*n-1]
   * DE2 = work[2*n:3*n-1] strting at begin */
  off_DE2 = 2*n;
  
  /* Shift eigenvalues to be consistent with dqds 
   * and compute eigenvalues of SHIFTED matrix */
  for (i=wbgn; i<=wend; i++) {
    W[i]    -= sigma;
    Werr[i] += fabs(W[i])*DBL_EPSILON;
  }
  
  /* Compute DE2 at store it in work[begin+2*n:end-1+2*n] */
  for (i=ibgn; i<iend; i++)
    work[i+off_DE2] = D[i]*E[i]*E[i];
  
  nthreads = max_nthreads;
  while (nthreads > 1 && mbl/nthreads < MIN_BISEC_CHUNK)
    nthreads--;
  
  if (nthreads > 1) {
    
    threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
    assert(threads != NULL);
    
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    
    rf_begin = wbgn;
    
    for (i=1; i<nthreads; i++) {

      chunk  = mbl/nthreads + (i < mbl%nthreads);
      rf_end = rf_begin + chunk - 1;

      arg = create_auxarg2(i, nbl, rf_begin, rf_end, spdiam, 
			   &D[ibgn], &work[ibgn+off_DE2], 
			   Wstruct, tolstruct);
      
      info = pthread_create(&threads[i], &attr, eigval_refine,
			    (void *) arg);
      assert(info == 0);
      
      rf_begin = rf_end+1;
    }
    
    rf_end = wend;

    arg = create_auxarg2(0, nbl, rf_begin, rf_end, spdiam, 
			 &D[ibgn], &work[ibgn+off_DE2], 
			 Wstruct, tolstruct);
    
    status = eigval_refine((void *) arg);
    assert(status == NULL);
    
    for (i=1; i<nthreads; i++) {
      info = pthread_join(threads[i], &status);
      assert(info == 0 && status == NULL);
    }
    
    /* update right gap at the splitting points */
    rf_begin = wbgn;
    for (i=1; i<nthreads; i++) {
      
      chunk  = mbl/nthreads + (i < mbl%nthreads);
      rf_end = rf_begin + chunk - 1;
      
      /* not Wgab[i] = fmax(Wgap[i], X), below is what dlarrb does */
      Wgap[rf_end] = (W[rf_end + 1] - Werr[rf_end + 1])
	             - (W[rf_end] + Werr[rf_end]);
      
      rf_begin = rf_end + 1;
    }
    
    Wgap[wend] = fmax(0.0, (*vup-sigma) - (W[wend] +  Werr[wend]) ); 
        
    free(threads);
    pthread_attr_destroy(&attr);
    
  } else {
    /* do refinement sequential */
    
    offset = ilow-1;
    
    /* refine eigenvalues found by dlarrd for ilow:iupp */
    dlarrb_(&nbl, &D[ibgn], &work[ibgn+off_DE2], &ilow,
	    &iupp, &rtol1, &rtol2, &offset, &W[wbgn], &Wgap[wbgn],
	    &Werr[wbgn], work, iwork, &pivmin, &spdiam, &nbl,
	    &info);
    assert(info == 0);
    /* needs work of dim(2*n) and iwork of dim(2*n) */
    
    /* dlarrb computes gaps correctly, but not last one
     * record distance to gu/vu */
    Wgap[wend] = fmax(0.0, (*vup-sigma) - (W[wend] + Werr[wend]) ); 
        
  } /* end parallel if worth doing so */

  return(0);
}




static inline 
aux2_t *create_auxarg2(int tid, int nbl, int rf_begin, int rf_end, 
		       double spdiam, double *D, double *DE2, 
		       val_t *Wstruct, tol_t *tolstruct)
{
  aux2_t *arg;
  
  arg = (aux2_t *) malloc(sizeof(aux2_t));
  assert(arg != NULL);

  arg->tid       = tid;
  arg->nbl       = nbl;
  arg->rf_begin  = rf_begin;
  arg->rf_end    = rf_end;
  arg->spdiam    = spdiam;
  arg->D         = D;
  arg->DE2       = DE2;
  arg->Wstruct   = Wstruct;
  arg->tolstruct = tolstruct;

  return(arg);
}


static inline 
void retrieve_auxarg2(aux2_t *arg, int *tid, int *nbl, int *rf_begin, 
		      int *rf_end, double *spdiam, double **D, 
		      double **DE2, val_t **Wstruct, tol_t **tolstruct)
{
  *tid       = arg->tid;
  *nbl       = arg->nbl;
  *rf_begin  = arg->rf_begin;
  *rf_end    = arg->rf_end;
  *spdiam    = arg->spdiam;
  *D         = arg->D;
  *DE2       = arg->DE2;
  *Wstruct   = arg->Wstruct;
  *tolstruct = arg->tolstruct;

  free(arg);
}


static void *eigval_refine(void *argin)
{
  /* Inputs */
  int    tid;
  int    nbl;
  int    rf_begin;
  int    rf_end;
  double *D;
  double *DE2;
  double spdiam;
  val_t  *Wstruct;
  tol_t  *tolstruct;

  /* Others */
  int    info;
  int    offset;
  double *work;
  int    *iwork;
  double savegap;
  
  retrieve_auxarg2((aux2_t*) argin, &tid, &nbl, &rf_begin, &rf_end, 
		   &spdiam, &D, &DE2, &Wstruct, &tolstruct);

  double *W      = Wstruct->W;
  double *Werr   = Wstruct->Werr;
  double *Wgap   = Wstruct->Wgap;
  int    *Windex = Wstruct->Windex;

  double rtol1   = tolstruct->rtol1;
  double rtol2   = tolstruct->rtol2;
  double pivmin  = tolstruct->pivmin;
    
  /* malloc work space */
  work = (double *) malloc( 2*nbl * sizeof(double) );
  assert(work != NULL);
  
  iwork = (int *) malloc( 2*nbl * sizeof(int) );
  assert(iwork != NULL);

  /* special case of only one eigenvalue */
  if (rf_begin == rf_end) {
    savegap = Wgap[rf_begin];
    Wgap[rf_begin] = 0.0;
  }

  offset = Windex[rf_begin] - 1;
  
  /* call bisection routine to refine the eigenvalues */
  dlarrb_(&nbl, D, DE2, &Windex[rf_begin], &Windex[rf_end],
	  &rtol1, &rtol2, &offset, &W[rf_begin], &Wgap[rf_begin],
	  &Werr[rf_begin], work, iwork, &pivmin, &spdiam,
	  &nbl, &info);
  assert(info == 0);

  if (rf_begin == rf_end) 
    Wgap[rf_begin] = savegap;

  free(work);
  free(iwork);

  return(NULL);
}






/* Finds the initial approximations of the eigenvalues,
 * requiering work dim(2*n) and iwork(2*n) */
static inline 
int find_eigval_irreducible(int max_nthreads, int n, int il, int iu, 
			    double *restrict D, double *restrict E, double *restrict E2, 
			    double *restrict W, double *restrict Werr,  
			    int *restrict Windex, double *restrict gersch, 
			    tol_t *tolstruct, double *work, int *iwork)
{
  /* Inputs */
  int    nvals            = iu - il + 1;

  /* Others */
  int            IONE=1;
  int            i, info;
  int            nthreads;
  pthread_t      *threads;
  pthread_attr_t attr;
  aux1_t         *arg;
  int            my_il, my_iu;
  int            chunk;
  void           *status;
  int            *tmp_iblock, *tmp_isplit;  
  double    dummy;
  double    vl, vu;
  int           m;

  nthreads = max_nthreads;
  while (nthreads > 1 && nvals/nthreads < MIN_BISEC_CHUNK)
    nthreads--;
  
  if (nthreads > 1) {
    /* Do computation in parallel */
    
    threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
    assert(threads != NULL);
    
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    
    /* The work will be split in the way that each threads finds
     * values 'my_il' to 'my_iu' and places them in work[0:n-1]; the
     * correspondind semi-width errors in work[n:2*n-1]; the blocks
     * they belong in iwork[0:n-1]; and their indices in
     * iwork[n:2*n-1]; */
    my_il = il;

    for (i=1; i<nthreads; i++) {

      chunk = nvals / nthreads + (i < nvals % nthreads);
      my_iu = my_il + chunk - 1;
      
      arg = create_auxarg1(i, il, iu, my_il, my_iu, n, D, E, E2, W, Werr, 
			   Windex, gersch, tolstruct);

      info = pthread_create(&threads[i], &attr, eigval_approx,
  			                     (void *) arg);
      assert(info == 0);
      
      my_il = my_iu + 1;
    }
    my_iu = iu;
    
    arg = create_auxarg1(0, il, iu, my_il, my_iu, n, D, E, E2, W, Werr, 
			 Windex, gersch, tolstruct);

    status = eigval_approx((void *) arg);
    assert(status == NULL);
    
    /* Join threads */
    for (i=1; i<nthreads; i++) {
      info = pthread_join(threads[i], &status);
      assert(info == 0 && status == NULL);
    }

    pthread_attr_destroy(&attr);
    free(threads);
    
  } else { 
    /* Do computation sequentially */
    
    tmp_iblock = (int *) malloc(n * sizeof(int));
    assert(tmp_iblock != NULL);

    tmp_isplit = (int *) malloc(n * sizeof(int));
    assert(tmp_isplit != NULL);
    
    tmp_isplit[0] = n;
   
    dlarrd_("I", "B", &n, &dummy, &dummy, &il, &iu, gersch, 
	    &tolstruct->bsrtol, D, E, E2, &tolstruct->pivmin, &IONE, 
	    tmp_isplit, &m, W, Werr, &vl, &vu,
  	    tmp_iblock, Windex, work, iwork, &info);
    assert(info == 0);

    free(tmp_iblock); 
    free(tmp_isplit); 
  } /* Parallel bisection or sequential */

  return(0);
}




static inline 
void *eigval_approx(void *argin)
{
  /* Inputs */
  int    tid, my_il, my_iu;
  tol_t  *tolstruct;
  double *W, *Werr;
  int    *Windex;

  int        n;
  double *D, *E, *E2;
  int        il, iu;
  double *gersch;

  /* Others */
  int    IONE=1;
  int    info, nvals;
  double *W_tmp, *Werr_tmp;
  double *work;
  int    *Windex_tmp, *iblock_tmp, *isplit_tmp;
  int    *iwork;
  double wl, wu, dummy;

  retrieve_auxarg1((aux1_t*) argin, &tid, &il, &iu, &my_il, &my_iu, &n, 
		   &D, &E, &E2, &W, &Werr, &Windex, &gersch, &tolstruct); 
  
  W_tmp = (double *) malloc( n * sizeof(double) );
  assert(W_tmp != NULL);

  Werr_tmp = (double *) malloc( n * sizeof(double) );
  assert(Werr_tmp != NULL);

  Windex_tmp = (int *) malloc( n * sizeof(int) );
  assert(Windex_tmp != NULL);

  iblock_tmp = (int *) malloc( n * sizeof(int) );
  assert(iblock_tmp != NULL);

  isplit_tmp = (int *) malloc( n * sizeof(int) );
  assert(isplit_tmp != NULL);
  isplit_tmp[0] = n;

  work = (double *) malloc( 4*n * sizeof(double) );
  assert (work != NULL);

  iwork = (int *) malloc( 3*n * sizeof(int) );
  assert (iwork != NULL);

  dlarrd_("I", "B", &n, &dummy, &dummy, &my_il, &my_iu, gersch,
  	  &tolstruct->bsrtol, D, E, E2, &tolstruct->pivmin, &IONE, 
	  isplit_tmp, &nvals, W_tmp, Werr_tmp, &wl, &wu, iblock_tmp, 
	  Windex_tmp, work, iwork, &info);
  assert(info == 0);
  assert(nvals = my_iu - my_il + 1);

  memcpy(&W[my_il - il], W_tmp, nvals * sizeof(double));
  memcpy(&Werr[my_il - il], Werr_tmp,   nvals * sizeof(double));
  memcpy(&Windex[my_il - il], Windex_tmp, nvals * sizeof(int));

  /* if (my_il == *ilp)      *vlp = wl; */
  /* else if (my_iu == *iup) *vup = wu; */
  
  free(W_tmp);
  free(Werr_tmp);
  free(Windex_tmp);
  free(iblock_tmp);
  free(isplit_tmp);
  free(work);
  free(iwork);

  return(NULL);
}
