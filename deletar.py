import os,sys

path = '/home/marquesleandro'

files = filter(os.path.isdir,os.listdir(os.curdir))

ww = 1
for name in files:
 os.chdir(name)
 
 a2 = os.listdir(os.curdir)

 if 'malha_convection.msh' in a2:
  print(name)
  break
  ww = 0

 if a2 != []:
  a1 = filter(os.path.isdir,os.listdir(os.curdir))
  for name1 in a1:
   os.chdir(name1)
   a3 = os.listdir(os.curdir)

   if 'malha_convection.msh' in a3:
    print(name)
    break
    ww = 0

   else:
    os.chdir('../')
  os.chdir(path)

 if ww == 0:
  break


