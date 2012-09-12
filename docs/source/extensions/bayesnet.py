# -*- coding: utf-8 -*-

# Copyright (c) 2012 Jaakko Luttinen

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.

#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY CHRISTOPH RELLER ''AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL CHRISTOPH RELLER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Christoph Reller.

from tikz import TikzDirective, tikz_role
from sphinx.ext.autodoc import ViewList
import os

BAYESNET_LIBS = 'shapes, fit, chains, arrows'

BAYESNET_DEFS = r'''
\tikzstyle{latent} = [circle,fill=white,draw=black,inner sep=1pt]
\tikzstyle{obs} = [latent,fill=gray!25]
\tikzstyle{const} = [rectangle, inner sep=0pt, node distance=1]
\tikzstyle{factor} = [rectangle, fill=black,minimum size=5pt, inner
sep=0pt, node distance=0.4]
\tikzstyle{det} = [latent, diamond]
'''

bnfile = open(os.path.join(os.path.abspath('.'), 'source/tikzlibrarybayesnet.code.tex'))
BAYESNET_DEFS = bnfile.read()
bnfile.close()

def bayesnet_role(role, rawtext, text, lineno, inliner, option={}, content=[]):
    return tikz_role(role, rawtext, text, lineno, inliner, option=option, content=content)

class BayesNetDirective(TikzDirective):
    def run(self):
        # Add TikZ libraries
        #self.options['libs'] = self.options.get('libs', '') + ',' + BAYESNET_LIBS
        # Run TikZ node
        (node,) = super(BayesNetDirective, self).run()
        # Add TikZ-BayesNet definitions
        node['tikz'] = BAYESNET_DEFS + node['tikz']
        print(BAYESNET_DEFS)
        return [node]

def setup(app):
    app.add_role('tikz', bayesnet_role)
    app.add_directive('bayesnet', BayesNetDirective)
