..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under Version 3.0 of the GNU General Public License.
   See LICENSE for a text of the license.

   This file is part of BayesPy.

   BayesPy is free software: you can redistribute it and/or modify it under the
   terms of the GNU General Public License version 3 as published by the Free
   Software Foundation.

   BayesPy is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
   A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along with
   BayesPy.  If not, see <http://www.gnu.org/licenses/>.


Implementing inference engines
==============================

Currently, only variational Bayesian inference engine is implemented.  This
implementation is not very modular, that is, the inference engine is not well
separated from the model construction.  Thus, it is not straightforward to
implement other inference engines at the moment.  Improving the modularity of
the inference engine and model construction is future work with high priority.
In any case, BayesPy aims to be an efficient, simple and modular Bayesian
package for variational inference at least.
