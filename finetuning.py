# this script is adapted from https://github.com/cltl/a-proof-zonmw/blob/main/clf_domains/train_model.py
import pandas as pd
from simpletransformers.classification import ClassificationModel
import logging

def finetune_model(trainpath:str) -> None:
    '''
    This function fine-tunes the MedRoBERTA.nl model for relative temporal extraction task.

    :param trainpath: str path to the training dataset
    :return None:
    '''
    # logging
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.WARNING)

    # load data and rename gold labels to numerical simplified ones
    train_data = pd.read_csv(trainpath,sep='\t')
    train_data = train_data.loc[:,['text','rel_time']]
    train_data.rename(columns={'rel_time':'labels'},inplace=True)
    mapping = {'past': 0, 'now': 1, 'future': 2}
    train_data.labels.replace(mapping,inplace=True)

    # model arguments used
    # note # multiprocessing needs to be turned off for working with the remote server
    model_args = {
        "max_seq_length": 512,
        "manual_seed": 27,
        "save_steps": 500,
        "process_count": 4,
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
        "dataloader_num_workers": 8
        }

    # model type, path on the server, arguments, and use of GPUs
    model = ClassificationModel(
        "roberta",
        "../../medroberta",
        num_labels=3,
        args=model_args,
        use_cuda=False,
    )

    # fine-tuning for the task
    model.train_model(train_data)

def predict_results(datapath:str) -> None:
    '''
    This function makes predictions of relative temporal labels by the final fine-tuned model 
    and saves the predictions as a column in a new .TSV file.

    :param datapath: str path to the dataset to make predictions on
    :return None:
    '''

    # read in the dataset
    df = pd.read_csv(datapath, sep='\t',dtype='string')

    # get the fine-tuned model from a local folder
    model = ClassificationModel(
        "roberta",
        "outputs/final_model",
        use_cuda=False
    )

    # extract the instances to predict on
    txt = df['text'].to_list()
    print("Got list of sentences!")
    print("Generating predictions. This might take a while...")
    predictions, _ = model.predict(txt)

    # write in the predictions into a new column and save to a new .TSV file
    df['pred_medroberta'] = predictions
    stripped_path = datapath.removesuffix('.tsv')
    outpath = f"{stripped_path}_medroberta_predictions.tsv"
    df.to_csv(outpath,sep='\t',index=False)
    print(f"A column with predictions was added.\nThe updated df is saved: {outpath}")