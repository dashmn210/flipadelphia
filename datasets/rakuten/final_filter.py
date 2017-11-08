

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False



f = open('final.tsv')

for l in f:
    parts = l.strip().split('\t')

    if len(parts) != 6:
        continue

    if not is_number(parts[1]) or not is_number(parts[3]):
        continue

    print '\t'.join(parts)
