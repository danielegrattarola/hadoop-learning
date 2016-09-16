import numpy as np

def flat2gen(alist):
  for item in alist:
    if isinstance(item, list) or isinstance(item, np.ndarray):
      for subitem in item: yield subitem
    else:
      yield item