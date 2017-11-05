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

    if not is_number(parts[9]) or \
       not is_number(parts[10]) or \
       not is_number(parts[11]) or \
       not is_number(parts[12]) or \
       not is_number(parts[24]) or \
       not is_number(parts[25]) or \
       not is_number(parts[26]) or \
       not is_number(parts[27]):

       continue

# 10, 11, 12, 13, 25, 26, 27, 28 should be numberso
# (we will control for average price rating, since that's a better indicator of high vs low end)

#    if not is_number(parts[23]):
#        continue
    print line
