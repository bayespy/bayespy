
/* ========================================================================== */
/* === sparse_distance.c ==================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Copyright (C) 2012 Jaakko Luttinen
 *
 * cholmod_spdist.c is licensed under Version 3.0 of the GNU General
 * Public License. See LICENSE for a text of the license.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * This file is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as 
 * published by the Free Software Foundation.
 *
 * This file is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this file.  If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 *
 * Computes a sparse squared distance matrix. Squared distances that
 * are larger than a given threshold are ignored in the sparsity
 * structure. Zero distances are marked with explicit zeros. If X2 is
 * NULL, then a symmetric distance matrix (in lower triangular form)
 * is computed for X1.
 *
 * -------------------------------------------------------------------------- */

#include <stdlib.h>
#include <stdio.h>
#include "sparse_distance.h"

#define EXPAND_TRIPLET_IF_NEEDED(Dx,Dij,nzmax,nz) \
{ \
    if (nz == nzmax) \
    { \
        nzmax *= 2 ; \
        Dij = realloc(Dij, 2*nzmax*sizeof(int)) ; \
        Dx = realloc(Dx, nzmax*sizeof(double)) ; \
    } \
}

#define FORM_HAS_DIAGONAL(f) (((f) == FULL) || ((f) == LOWER) || ((f) == UPPER))
#define FORM_HAS_LOWER(f) (((f) == LOWER) || ((f) == STRICTLY_LOWER) || ((f) == FULL))
#define FORM_HAS_UPPER(f) (((f) == UPPER) || ((f) == STRICTLY_UPPER) || ((f) == FULL))

/* Compute squared Euclidean distance of two vectors */
inline double sqeuclidean_distance
(
    const double *x1,
    const double *x2,
    int n
)
{
    int i ;
    double d, d2 = 0 ;
    for (i = 0; i < n; i++)
    {
        d = x1[i]-x2[i] ;
        d2 += d*d ;
    }
    return d2 ;
}


/* ========================================================================== */
/* === sppdist_sqeuclidean ================================================== */
/* ========================================================================== */

void sppdist_sqeuclidean
(
    /* ---- input ---- */
    const double *X,  /* input matrix, m-by-n in row-major form */
    int m, /* m vectors */
    int n, /* n-dimensional vectors */
    double threshold,   /* distance threshold */
    int form, /* full / lower / strictly lower */
    /* ---- output ---- */
    double **out_Dx, /* distances */
    int **out_Dij, /* row and column indices */
    int *out_nzmax /* number of non-zeros, i.e., length of Di and Dx */
)
{

    int i, j ;
    int nz, nzmax;
    double d ;
    const double *x1, *x2 ;
    int *Dij ;
    double *Dx ;

    // Initial size for the number of non-zero elements
    nzmax = 2*m ;

    // Allocate initial memory
    Dij = realloc(*out_Dij, 2*nzmax*sizeof(int)) ;
    Dx = realloc(*out_Dx, nzmax*sizeof(double)) ;
    
    // Number of non-zeros thus far
    nz = 0 ;

    for (j = 0; j < m; j++)
    {

        if (FORM_HAS_DIAGONAL(form)) 
        {
            Dij[2*nz] = j ;
            Dij[2*nz+1] = j ;
            Dx[nz] = 0.0 ;
            nz++ ;
            EXPAND_TRIPLET_IF_NEEDED(Dx, Dij, nzmax, nz) ;
        }
            
        for (i = j+1; i < m; i++)
        {
            // Compute distance between X1[:,i] and X2[:,j]
            x1 = X + i*n ;
            x2 = X + j*n ;
            d = sqeuclidean_distance(x1, x2, n) ;
            
            // Store distances that are small enough
            if (d <= threshold)
            {
                // Store strictly lower triangular part
                if (FORM_HAS_LOWER(form))
                {
                    Dij[2*nz] = i ;
                    Dij[2*nz+1] = j ;
                    Dx[nz] = d ;
                    nz++ ;
                    EXPAND_TRIPLET_IF_NEEDED(Dx, Dij, nzmax, nz) ;
                }
                
                if (FORM_HAS_UPPER(form))
                {
                    // Store strictly upper triangular part
                    Dij[2*nz] = j ;
                    Dij[2*nz+1] = i ;
                    Dx[nz] = d ;
                    nz++ ;
                    EXPAND_TRIPLET_IF_NEEDED(Dx, Dij, nzmax, nz) ;
                }
            }
            
        }
    }

    // Store the matrix compactly
    nzmax = nz ;
    Dij = realloc(Dij, 2*nzmax*sizeof(int)) ;
    Dx = realloc(Dx, nzmax*sizeof(double)) ;

    // Set output variables
    *out_Dij = Dij ;
    *out_Dx = Dx ;
    *out_nzmax = nzmax ;

    return ;

}


void spcdist_sqeuclidean
(
    /* ---- input ---- */
    const double *X1,  /* input matrix, m-by-n in row-major form */
    int m1, /* m1 vectors */
    const double *X2,  /* input matrix, m-by-n in row-major form */
    int m2, /* m2 vectors */
    int n, /* n-dimensional vectors */
    double threshold,   /* distance threshold */
    /* ---- output ---- */
    double **out_Dx, /* distances */
    int **out_Dij, /* row and column indices */
    int *out_nzmax /* number of non-zeros */
)
{

    int i, j ;
    int nz, nzmax;
    double d ;
    const double *x1, *x2 ;
    int *Dij ;
    double *Dx ;

    // Initial size for the number of non-zero elements
    nzmax = 2 * (m1 > m2 ? m1 : m2) ;

    // Allocate initial memory
    Dij = realloc(*out_Dij, 2*nzmax*sizeof(int)) ;
    Dx = realloc(*out_Dx, nzmax*sizeof(double)) ;
    
    // Number of non-zeros thus far
    nz = 0 ;

    for (i = 0; i < m1; i++)
    {
        for (j = 0; j < m2; j++)
        {
            // Compute distance between X1[:,i] and X2[:,j]
            x1 = X1 + i*n ;
            x2 = X2 + j*n ;
            d = sqeuclidean_distance(x1, x2, n) ;
            
            // Store distances that are small enough
            if (d <= threshold)
            {
                Dij[2*nz] = i ;
                Dij[2*nz+1] = j ;
                Dx[nz] = d ;
                nz++ ;
                EXPAND_TRIPLET_IF_NEEDED(Dx, Dij, nzmax, nz) ;
            }
            
        }
    }

    // Store the matrix compactly
    nzmax = nz ;
    Dij = realloc(Dij, 2*nzmax*sizeof(int)) ;
    Dx = realloc(Dx, nzmax*sizeof(double)) ;

    // Set output variables
    *out_Dij = Dij ;
    *out_Dx = Dx ;
    *out_nzmax = nzmax ;

    return ;

}
