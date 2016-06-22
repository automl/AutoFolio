#script(python)

import gringo

ts = {}
def insert(i,s,t):
  key = str(s)
  if not ts.get(key):
    ts[key] = []
  ts[key].append([i,t])
  return 1

def order(s):
  key = str(s)
  if not ts.get(key):
    ts[key] = []
  ts[key].sort(key=lambda x: int(x[1]))
  p = None
  r = []
  for i, v in ts[key]:
    if p:
      r.append((p,i))
    p = i
  return r

#end.


