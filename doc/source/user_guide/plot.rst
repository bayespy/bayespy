..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under Version 3.0 of the GNU General Public
   License. See LICENSE for a text of the license.

   This file is part of BayesPy.

   BayesPy is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 3 as
   published by the Free Software Foundation.

   BayesPy is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.


Examining the results
---------------------

After the results have been obtained, it is important to be able to
examine the results easily.  ``show`` method prints the approximate
posterior distribution of the node.  Also, ``get_moments`` can be used
to obtain the sufficient statistics of the node.

.. todo::

   In order to examine the results more carefully, ``get_parameters``
   method should return the parameter values of the approximate
   posterior distribution.  The user may use these values for
   arbitrarily complex further analysis.


