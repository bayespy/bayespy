# -*- coding: utf-8 -*-

# Copyright (c) 2012 Jaakko Luttinen


from tikz import TikzDirective, tikz_role
from sphinx.ext.autodoc import ViewList
import os

BAYESNET_LIBS = 'shapes, fit, chains, arrows'

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
