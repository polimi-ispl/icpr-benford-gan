import jpeg
import sys
import random
a = jpeg.Jpeg(sys.argv[1])

def swap(list,a,b):
  tmp = list[a]
  list[b] = list[a]
  list[a] = tmp

comps = a.component_count
for comp in range(comps):
  blocks = []
  xmax, ymax = a.getcomponentdimensions(comp)
  for y in range(ymax):
    for x in range(xmax):
      blocks.append(a.getblock(x,y,comp))

  random.shuffle(blocks)

  i = 0
  for y in range(ymax):
    for x in range(xmax):
      a.setblock(x,y,comp,blocks[i])
      i += 1
a.write(sys.argv[2])
