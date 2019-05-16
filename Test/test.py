def conv(h, fh, w, fw, s):
    return ((h-fh)/s)+1, ((w-fw)/s)+1

def pooling(h, w, sh, sw):
    return h/sh, w/sw


h, w = conv(600, 9, 450, 6, 1)
print(h, w)     # 592,  445


h, w = conv(h, 9, w, 6, 1)
print(h, w)   # 584, 440

h, w = pooling(h, w, 2, 2)
print(h, w)     #292, 220
print('1st conv layer')

h, w = conv(h, 7, w, 3, 1)
print(h, w)    # 286, 218

h, w = conv(h, 7, w, 3, 1)
print(h, w)    # 280, 216

h, w = pooling(h, w, 2, 2)
print(h, w)  # 140, 108
print(" 2nd conv layer " )

h, w = conv(h, 5, w, 3, 1)
print(h, w)
h, w = conv(h, 5, w, 3, 1)
print(h, w)
h, w = pooling(h, w, 2, 2)
print(h, w)
print(" 3rd conv layer " )




