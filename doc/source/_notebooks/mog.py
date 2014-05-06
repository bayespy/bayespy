
# coding: utf-8

## Example: Mixture of Gaussians

# Do some stuff:

# In[2]:

from bayespy.nodes import Dirichlet
alpha = Dirichlet([1e-3, 1e-3, 1e-3])
print(alpha._message_to_child())


# Nice!
