

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

input = open('inputs')
output = open('test')


for il, ol in zip(input, output):
    if is_number(ol.split('\t')[2]):
        print il.strip() + '\t' + ol.strip()

