/* ========================================================================== */
/* === sparse_distance.h ==================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Copyright (C) 2012 Jaakko Luttinen
 *
 * cholmod_spdist.h is licensed under Version 3.0 of the GNU General
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

#ifndef SPARSE_DISTANCE_H
#define SPARSE_DISTANCE_H

#define FULL 0
#define LOWER 1
#define UPPER 2
#define STRICTLY_LOWER 3
#define STRICTLY_UPPER 4

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
    int *out_nzmax /* number of non-zeros */
) ;

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
) ;

#endif
