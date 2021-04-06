import jpeg
import sys
a = jpeg.Jpeg(sys.argv[1])

count = a.component_count
for comp in range(count):
  xmax, ymax = a.getcomponentdimensions(comp)
  for y in range(ymax):
    for x in range(xmax):
      block = a.getblock(x,y,comp)
      for i in range(len(block)):
        block[i] = 0
      a.setblock(x,y,comp,block)
a.write(sys.argv[2])
