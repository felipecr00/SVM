\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.1 u0 + 0.1 u1 + 0.1 u2 + 0.1 v0 + 0.1 v1 + 0.1 v2 + [ 2 a0 ^2 + 2 a1 ^2
   ] / 2 
Subject To
 x0: a1 - beta + u0 >= 1
 x1: a0 + 2 a1 - beta + u1 >= 1
 x2: 3 a0 + 4 a1 - beta + u2 >= 1
 y0: a0 + a1 - beta - v0 <= -1
 y1: 2 a0 + 2 a1 - beta - v1 <= -1
 y2: 4 a0 + 4 a1 - beta - v2 <= -1
Bounds
 a0 free
 a1 free
 beta free
End
