#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

import numpy as np
from sympy import *
init_printing()


a,b,c,d,e,f = symbols('a b c d e f')
K = IndexedBase('K')

Nfaces = 3
Nsubfaces = 2

Ntiles = Nfaces*Nsubfaces

perm_c = np.zeros((Nfaces,Nfaces),int)
perm_e = np.zeros((Nfaces,Nfaces),int)
sig = np.array([[0,1],[1,0]])

perm_c[1:,:-1] = np.eye(Nfaces-1,Nfaces-1)
perm_c[0,-1] = 1

perm_e[2:,2:] = np.eye(Nfaces-2)
perm_e[:2,:2] = sig


tiles = np.arange(Ntiles).reshape(Nfaces,Nsubfaces)

print(tiles)

#for i in range(Nfaces):
#    for j in range(Nsubfaces):
#        tiles[i,j] = Nsubfaces*i+j

#kmat = np.zeros((Nsubfaces,Nsubfaces),object)


#for tile1 in tiles:
#    for tile2 in tiles:
#        np.array([[K[tile1[0],tile1[1],tile2[0],tile2[1]],
#                   K[tile1[0],tile1[1],tile2[1],tile2[0]]],
#                  [K[tile1[1],tile1[0],tile2[0],tile2[1]]
#                   K[tile1[1],tile1[0],tile2[1],tile2[0]]]])

## N=3 system

kmat = Matrix([[     0,     0,K[a,b],K[a,a],K[a,b],K[b,b]],
               [     0,     0,K[b,b],K[a,b],K[a,a],K[a,b]],
               [K[a,b],K[b,b],     0,     0,K[a,b],K[a,a]],
               [K[a,a],K[a,b],     0,     0,K[b,b],K[a,b]],
               [K[a,b],K[a,a],K[a,b],K[b,b],     0,     0],
               [K[b,b],K[a,b],K[a,a],K[a,b],     0,     0]])

pprint(kmat)


## N=4 system

kmat_face = Matrix([[K[a,b,f,e],K[a,b,e,f],K[a,b,d,c],K[a,b,c,d],K[a,b,b,a],K[a,b,a,b]],
                    [K[b,a,f,e],K[b,a,e,f],K[b,a,d,c],K[b,a,c,d],K[b,a,b,a],K[b,a,a,b]],
                    [K[e,f,f,e],K[e,f,e,f],K[e,f,d,c],K[e,f,c,d],K[e,f,b,a],K[e,f,a,b]],
                    [K[f,e,f,e],K[f,e,e,f],K[f,e,d,c],K[f,e,c,d],K[f,e,b,a],K[f,e,a,b]],
                    [K[c,d,f,e],K[c,d,e,f],K[c,d,d,c],K[c,d,c,d],K[c,d,b,a],K[c,d,a,b]],
                    [K[d,c,f,e],K[d,c,e,f],K[d,c,d,c],K[d,c,c,d],K[d,c,b,a],K[d,c,a,b]]])


aa = 0
bb = 1
cc = 2
dd = 3
ee = 4
ff = 5

ab = 6
ac = 7
ad = 8
ae = 9
af = 10

bc = 11
bd = 12
be = 13
bf = 14

cd = 15
ce = 16
cf = 17

de = 18
df = 19

ef = 20

kmat_face = Matrix([[K[af,be],K[ae,bf],K[ad,bc],K[ac,bd],K[ab,ab],K[aa,bb]],
                    [K[ae,bf],K[af,be],K[ac,bd],K[ad,bc],K[aa,bb],K[ab,ab]],
                    [K[ef,ef],K[ee,ff],K[cf,de],K[ce,df],K[af,be],K[ae,bf]],
                    [K[ee,ff],K[ef,ef],K[ce,df],K[cf,de],K[ae,bf],K[af,be]],
                    [K[cf,de],K[ce,df],K[cd,cd],K[cc,dd],K[ad,bc],K[ac,bd]],
                    [K[ce,df],K[cf,de],K[cc,dd],K[cd,cd],K[ac,bd],K[ad,bc]]])

pprint(kmat_face)
