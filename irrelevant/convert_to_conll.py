# based on https://github.com/cltl/a-proof-zonmw/blob/main/data_process/data_process_to_inception/text_to_conll.py by Jenia Kim
import pandas as pd 
import spacy, nl_core_news_sm
import csv 
## install on linux via: ##
# pip install https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.5.0/nl_core_news_sm-3.5.0-py3-none-any.whl
# pip install https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.5.0/nl_core_news_sm-3.5.0.tar.gz


def tsv_to_conll(inpath, outpath):
    with open(inpath, 'r',encoding='utf-8') as infile:
        csv_reader = csv.reader(infile, delimiter='\t')
        next(csv_reader)
        with open(outpath, 'w', encoding='utf-8', newline='') as outfile:
            if'test' in inpath:
                header = 'token\tpad_sent_id\tinstitution\tyear\tMDN\tNotitieID\tbatch\trel_time\n'
            else:
                header = 'token\tpad_sent_id\tinstitution\tyear\tMDN\tNotitieID\tbatch\trel_time\tnote_len\tUPOS\tXPOS\thead:dep\n'
            outfile.write(header)
            for line in csv_reader:
                ### EDIT ON VM : ADD FULL NOTE STUFF ###
                # pad_sent_id, institution, year, MDN, NotitieID, batch, labels, text, len_text, rel_time
                pad_sent_id = line[0]
                institution = line[1]
                year = line[2]
                mdn = line[3]
                note_id = line[4]
                batch = line[5]
                # icf_l_list = line[6] # maybe remove
                text = line[7]
                # len_text = line[8]
                rel_time = line[9]
                # tokenise text, get its upox, xpos, dependency label and head
                # add back the other columns
                nlp = nl_core_news_sm.load()
                doc = nlp(text)
                tab = '\t'
                if 'test' in inpath:
                    tokens = [token.text for token in doc]
                    rows = [f"{token}{tab}{pad_sent_id}{tab}{institution}{tab}{year}{tab}{mdn}{tab}{note_id}{tab}{batch}{tab}{rel_time}" for token in tokens]
                else:
                    note_date = line[10]
                    full_note_text = line[11]
                    doc_note = nlp(full_note_text)
                    note_len = str(len(list(doc_note.sents)))
                    tokens = [(token.text, token.pos_,token.tag_,token.dep_,token.head.text) for token in doc]
                    rows = [f"{token[0]}{tab}{pad_sent_id}{tab}{institution}{tab}{year}{tab}{mdn}{tab}{note_id}{tab}{batch}{tab}{rel_time}{tab}{note_len}{tab}{note_date}{tab}{token[1]}{tab}{token[2]}{tab}{token[4]}:{token[3]}" for token in tokens]
                conll = '\n'.join(rows)
                conll = conll+'\n'
                outfile.write(conll)
                

tsv_to_conll('../data/train.tsv','../data/train.conll')
tsv_to_conll('../data/dev.tsv','../data/dev.conll')
tsv_to_conll('../data/test.tsv','../data/test.conll')



