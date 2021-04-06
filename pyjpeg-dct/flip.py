import jpeg
import sys
import struct
a = jpeg.Jpeg(sys.argv[1])

def flipblock(block):
  tmp = str(block)
  for i in range(0, len(block), 4):
    short, = struct.unpack_from("h", tmp, i+2)
    struct.pack_into("h", block, i+2, -short)

for comp in range(a.component_count):
  xmax, ymax = a.getcomponentdimensions(comp)
  for y in range(ymax):
    for x in range((xmax+1)/2):
      block = a.getblock(x,y,comp)
      block2 = a.getblock(xmax-1 - x, y, comp)
      flipblock(block)
      flipblock(block2)
      a.setblock(xmax-1-x, y, comp, block)
      a.setblock(x,y,comp,block2)
a.write(sys.argv[2])
