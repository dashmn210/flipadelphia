import sys

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

for line in open(sys.argv[1]):
    line = line.strip()
    parts = line.split('\t')
    if not len(parts) == 28:
        continue
# 10, 11, 12, 13, 25, 26, 27, 28 should be numberso
# (we will control for average price rating, since that's a better indicator of high vs low end)

#    if not is_number(parts[23]):
#        continue
    print line
