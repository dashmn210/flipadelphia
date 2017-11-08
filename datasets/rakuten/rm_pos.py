


for l in open('descriptions.tsv'):
    parts = l.strip().split("\t")
    parts[0] = ' '.join(map(
            lambda x: x.split(':')[0], 
            parts[0].split()
    ))
    print '\t'.join(parts)









