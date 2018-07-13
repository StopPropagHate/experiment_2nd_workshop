import re
import os
import warnings

warnings.filterwarnings("ignore")

class SentiLex:

    flex = 'SentiLex-flex'
    lem = 'SentiLex-lem'
    lem_capture_pattern = '(.*)\.PoS=(N|Adj|V|IDIOM);TG=(.*);ANOT=(MAN|JALC);?(REV=Amb)?'
    data_path = os.path.expanduser('~/Desktop/experiment_2nd_workshop/pt_approach/lexicons')

    def __init__(self):
        self.lemas = {}
        lem_path = self.get_lexicon_path(SentiLex.lem)
        if (os.path.exists(lem_path)):
            with open(lem_path, 'r') as lem:
                lines = lem.readlines()
                for line in lines:

                    res = re.findall(SentiLex.lem_capture_pattern, line)
                    if res != []:
                        res = res.pop()
                        tags = res[2].split(';')
                        tags.pop(0)
                        pols = [int(pol.split('=')[1]) for pol in tags]
                        polAvg = sum(pols) / float(len(pols))
                        self.lemas[ res[0] ] = polAvg

    def get_sentiment_tweet(self,tokens):
        sentiment_scores = []
        for token in tokens:
            val = self.get_sentiment_lema(token)
            if val != None:
                sentiment_scores.append(val)
        if sentiment_scores != []:
            avg = sum(sentiment_scores) / float(len(sentiment_scores))
        else:
            avg=0
        return avg

    def get_sentiment_lema(self,lema):
        try:
            return self.lemas[lema]
        except KeyError:
            return None
    def get_lexicon_path(self,name):
        if (name == 'SentiLex-lem'):
            return self.data_path + '/SentiLex-lem-PT02.txt'
        if (name == 'SentiLex-flex'):
            return self.data_path + '/SentiLex-flex-PT02.txt'


test = False
if test:
    s = SentiLex()
    print(s.lemas)

    flex_capture_pattern ='(.*),(.*).PoS=(N|Adj|V|IDIOM);FLEX=(.*);TG=(.*);ANOT=(MAN|JALC)'
    sentilex_flex = s.get_lexicon_path(s.flex)

    with open(sentilex_flex,'r') as flex:
        lines = flex.readlines()
        for line in lines:
            res = re.findall(flex_capture_pattern,line)
            if(res == []):
                #print('\t ERROR PARSING \t',line)
                pass



