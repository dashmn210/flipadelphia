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
    if not len(parts) == 27:
        continue
#    if not is_number(parts[23]):
#        continue
    print line
