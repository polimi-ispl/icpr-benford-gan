from __future__ import print_function
def quitwith(s):
  print(s)
  raise Exception

print("importing jpeg")
import jpeg
print("loading a jpeg with a number")
try:
  a = jpeg.Jpeg(0)
  quitwith("this should have failed!")
except Exception as e:
  print(e)
print("loading a jpeg with a nonsense filename")
try:
  a = jpeg.Jpeg("")
  quitwith("this should have failed!")
except Exception as e:
  print(e)
print("loading a jpeg with a non-jpeg file")
try:
  a = jpeg.Jpeg(".")
  quitwith("this should have failed!")
except Exception as e:
  print(e)
print("loading a file with jpeg magic")
try:
  with open("magic", "wb") as magic:
    magic.write("ffd8".decode("hex"))
  a = jpeg.Jpeg("magic")
  quitwith("this should have failed!")
except Exception as e:
  print(e)
print("loading a real jpeg")
a = jpeg.Jpeg("out.jpg")
print("getting block")
block = a.getblock(0,0,0)
print("setting block with itself")
a.setblock(0,0,0,block)
print("setting block with too-small buffer")
try:
  a.setblock(0,0,0,block[:1])
  quitwith("this should have failed!")
except Exception as e:
  print(e)
print("setting block with None")
try:
  a.setblock(0,0,0,None)
  quitwith("this should have failed!")
except Exception as e:
  print(e)

print("writing jpeg")
a.write("out2.jpg")

def testblocks(a,b):
  err = False
  for comp in range(a.component_count):
    xmax, ymax = a.getcomponentdimensions(comp)
    for y in range(ymax):
      for x in range(xmax):
        if a.getblock(x,y,comp) != b.getblock(x,y,comp):
          print("block at {},{} in component {} doesn't match original".format(x,y,comp))
          err = True
  return err
b = jpeg.Jpeg("out2.jpg")
if not testblocks(a,b):
  print("all blocks from written jpeg match original")
