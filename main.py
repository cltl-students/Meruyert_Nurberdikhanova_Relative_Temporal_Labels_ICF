import pandas as pd 
import nl_core_news_sm, sys
from utils import *
from finetuning import *
from svm_exp import *
def main(argv=None) -> None:
    '''
    This function executes all the main parts of the experiment of this thesis. 

    :param argv: list of arguments specifying which parts of experiment to perform.

    :return None:
    '''
    if argv is None:
        argv = sys.argv

    get_data = argv[1]
    merge_data_by_gold = argv[2]
    create_split = argv[3]
    save_descriptives = argv[4]
    link_full_notes = argv[5]
    conll = argv[6]
    feature_engineer = argv[7]
    analyse_corpus = argv[8]
    svm_exp = argv[9]
    medroberta_clf_exp = argv[10]
    error_subdf = argv[11]
    
    ### ---------- steps to go through: ---------- ###
    # 1 # get data from pickled files in expr_sept 
    if get_data:
        train_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/train.pkl'
        test_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/test.pkl'
        dev_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/dev.pkl'
        get_rel_time_subset(train_path,'train')
        get_rel_time_subset(dev_path,'dev')
        get_rel_time_subset(test_path,'test')
    # 2 # merge data by gold labels
    if merge_data_by_gold:
        merge_data('background')
        merge_data('target')
        merge_data('current')
    # 3 # make train, dev, test split
    if create_split:
        back_train, back_dev, back_test = split_subgroup('background')
        tar_train, tar_dev, tar_test = split_subgroup('target')
        cur_train, cur_dev, cur_test = split_subgroup('current')
        merge_shuffle_split(back_train,tar_train,cur_train,'train')
        merge_shuffle_split(back_dev,tar_dev,cur_dev,'dev')
        merge_shuffle_split(back_test,tar_test,cur_test,'test')
    # 4 # get descriptive statistics of the data
    if save_descriptives:
        get_descriptives('batch')
        get_descriptives('year')
        get_descriptives('institution')
        get_descriptives('natural')
        get_descriptives('note_patient')
    # 5 # get full notes of the data & clean out duplicates
    if link_full_notes:
        train = pd.read_csv('../data/train.tsv',sep='\t',dtype='string')
        dev = pd.read_csv('../data/dev.tsv',sep='\t',dtype='string')
        save_full_notes(train,'train')
        save_full_notes(dev,'dev')
        test = pd.read_csv('../data/test.tsv',sep='\t',dtype='string')
        save_full_notes(test,'test')
    # 6 # convert from sentence to token level
    if conll:
        tsv_to_conll('../data/train_full_notes.tsv','../data/train.conll')
        tsv_to_conll('../data/dev_full_notes.tsv','../data/dev.conll')
        tsv_to_conll('../data/test_full_notes.tsv','../data/test.conll')
    # 7 #  and feature engineer for svm experiment
    if feature_engineer:
        nlp = nl_core_news_sm.load()
        write_features('../data/train.conll','train',nlp)
        write_features('../data/dev.conll','dev',nlp)
        write_features('../data/test.conll','test',nlp)
    # 8 # perform corpus analysis of each data split subset
    if analyse_corpus:
        corpus_analysis('../data/train_with_features.conll','train')
        corpus_analysis('../data/dev_with_features.conll','dev')
        corpus_analysis('../data/test_with_features.conll','test')
    # 9 # perform SVM experiments and get evaluation metrics
    if svm_exp:
        perform_svm_exp(1, 'dev')
        perform_svm_exp(2,'dev')
        perform_svm_exp(3,'dev')
        perform_svm_exp(4,'dev')
        perform_svm_exp(4,'test')
        make_heatmap('../data/test_svm_exp_4_predictions.tsv','svm','exp_4_test')
        make_heatmap('../data/test_svm_exp_4_predictions.tsv','svm','exp_4_test',True)
    # 10 # perform MedRoBERTa experiments and get evaluation metrics
    if medroberta_clf_exp:
        finetune_model('../data/train.tsv')
        predict_results('../data/dev_full_notes.tsv')
        predict_results('../data/test_full_notes.tsv')
        get_evaluation_metrics('../data/dev_full_notes_medroberta_predictions.tsv','medroberta','dev')
        get_evaluation_metrics('../data/test_full_notes_medroberta_predictions.tsv','medroberta','test')
        make_heatmap('../data/test_full_notes_medroberta_predictions.tsv','medroberta','test')
        make_heatmap('../data/test_full_notes_medroberta_predictions.tsv','medroberta','test',True)
    # 11 # isolate misclassifications and get corpus statistics for the error analysis
    if error_subdf:
        write_misclassified('../data/test_full_notes_medroberta_predictions.tsv','medroberta', 'test')
        write_misclassified('../data/test_svm_exp_4_predictions.tsv','svm','exp_4_test')
        error_subsets = ['now_future','now_past','future_now','future_past','past_future','past_now']
        for subset in error_subsets:
            get_error_conll(f'../data/misclassifications/{subset}_medroberta_test.tsv')
            get_error_conll(f'../data/misclassifications/{subset}_svm_exp_4_test.tsv')
            corpus_analysis(f'../data/misclassifications/{subset}_medroberta_test.conll',f'{subset}_medroberta')
            corpus_analysis(f'../data/misclassifications/{subset}_svm_exp_4_test.conll',f'{subset}_svm_exp_4')
            

main(['python',False,False,False,False,False,False,False,False,False,False,False])