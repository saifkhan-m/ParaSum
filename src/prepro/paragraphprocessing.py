import json
#from others.utils import clean

def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p,  encoding='utf8'))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])

        if flag:
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt
def load_paras(file,lower):
    source = []
    tgt = []
    flag = False
    with open(file, 'r') as story_file:
        paragraphs = story_file.read().split('\n\n')

        for para in paragraphs:
            if (para == '@highlight'):
                flag = True
                tgt.append([])

            if flag:
                tgt[-1].extend(para.split(' '))
            else:
                source.append(para.split(' '))


        print('jnbvdhj')
    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt
source, tgt = load_paras("C:/Users/khans/Documents/ber_hwr_hiwi/parasum/PreSumm/raw_data/sub100/00a2aef1e18d125960da51e167a3d22ed8416c09.story", True)
#source, tgt = load_json("C:/Users/khans/Documents/ber_hwr_hiwi/parasum/PreSumm/raw_data/sub100/00a2aef1e18d125960da51e167a3d22ed8416c09.story", True)