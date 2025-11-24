import os
import json
import pandas as pd
import xml.etree.ElementTree as ET
from Agent_pools import FinalPolarityAgentRaw, FinalPolarityAgentBench
from sklearn.metrics import f1_score, accuracy_score

tree_res_14_test = ET.parse('./RES14/Restaurants_Test_Gold_Implicit_Labeled.xml')
root_res_14_test = tree_res_14_test.getroot()
res_14_test = []
for sentence in root_res_14_test.findall('sentence'):
    sentence_id = sentence.get('id')
    sentence_text = sentence.find('text').text
    aspectTerms = sentence.findall('.//aspectTerm')
    for aspectTerm in aspectTerms:
        term = aspectTerm.get('term')
        polarity = aspectTerm.get('polarity')
        implicit = aspectTerm.get('implicit_sentiment')
        res_14_test.append({
            'sentence_id': sentence_id,
            'sentence_text': sentence_text,
            'aspect_term': term,
            'polarity': polarity,
            'implicit_sentiment': implicit
        })
df_res_14_test = pd.DataFrame(res_14_test)
df_res_14_test = df_res_14_test[df_res_14_test['polarity'] != 'conflict']

########################################
#           Data_Processing            #
########################################
with open('./RES14/res_14_test.json', 'r') as file:
    res14 = json.load(file)
dataset_output = []
for i in range(len(res14)):
    dataset = {'comment': res14[i]['sentence_text'],
               'aspect_name': res14[i]['aspect_term'],
               'polarity': res14[i]['polarity']}
    dataset_output.append(dataset)
with open('./res14_prompt_output/english_few_shot.json', 'r') as file:
    few_shots_pools = json.load(file)

cot_final = """
- Q1: Does the comment directly mention {aspect_name} and use strong positive or negative sentiment words?
- Q2: Even if the comment does not directly mention {aspect_name}, is there a description that can indirectly infer positive or negative sentiment about {aspect_name}?
- Q3: Does the comment provide a contextually positive or negative description of {aspect_name}?
- Q4: Does the comment only provide a neutral description of {aspect_name}, without any clear positive or negative sentiment?
- Q5: Based on the answers to the previous four questions, what do you think is the overall sentiment polarity towards {aspect_name} in the restaurant review?
"""

cot_raw = """
what's sentiment polarity towards {aspect_name} in the restaurant review?
"""

if __name__ == '__main__':

    ########################################
    #              ExpertABSA              #
    ########################################
    for i in range(0, 5):
        path = f'./testing_res_14/main_result_and_ablation_study_english/expertabsa_llama_epochs{i}'
        os.makedirs(path, exist_ok=False)
        model_initialise = FinalPolarityAgentBench(
                                              cot_questions_final=cot_final,
                                              testing_datasets=dataset_output[0:10],
                                              fetch_k=8,
                                              total_K=20,
                                              prompt_path='./prompt_templates_res14/english_version'
                                                          '/res14_polarity_agent_eng_prompt.yaml',
                                              few_shot_pools=few_shots_pools)
        prompt = model_initialise.prompt_evaluate()
        result_df, metrics_test = model_initialise.evaluate_local(thinking_mode=False)
        df_output = result_df
        df_output['prompt'] = prompt
        df_output.to_excel(os.path.join(path, f'expertabsa_result_{i}.xlsx'))
        print(metrics_test)
        with open(os.path.join(path, f'expertabsa_metrics_{i}.txt'), 'w') as file:
            file.write(str(metrics_test) + '\n')
            file.write(str(cot_final))
        # calculate metrics
        df_result = pd.read_excel(os.path.join(path, f'expertabsa_result_{i}.xlsx'))
        df_result = df_result.rename(columns={'comment': 'sentence_text', 'aspect_name': 'aspect_term'})
        df_result = df_result.drop(columns='Unnamed: 0')
        df_result = df_result[['aspect_term', 'sentence_text', 'pred_polarity']]
        df_res_14_test = df_res_14_test.reset_index()
        df_res_14_test = df_res_14_test.drop(columns=['index'])
        combined_df = pd.merge(df_res_14_test, df_result, left_index=True, right_index=True, how='inner')
        implicit_df = combined_df[combined_df['implicit_sentiment'] == 'True']
        implicit_df['pred_polarity'].fillna("neutral", inplace=True)
        explicit_df = combined_df[combined_df['implicit_sentiment'] == 'False']
        explicit_df['pred_polarity'].fillna("neutral", inplace=True)
        metrics_acc_implicit = accuracy_score(y_true=implicit_df["polarity"], y_pred=implicit_df["pred_polarity"])
        metrics_f1_implicit = f1_score(y_true=implicit_df["polarity"], y_pred=implicit_df["pred_polarity"],
                                       average='macro')
        print(f"expert_absa_round_{i}")
        print(f"acc_implicit:{metrics_acc_implicit},f1_implicit:{metrics_f1_implicit}")
        metrics_acc_explicit = accuracy_score(y_true=explicit_df["polarity"], y_pred=explicit_df["pred_polarity"])
        metrics_f1_explicit = f1_score(y_true=explicit_df["polarity"], y_pred=explicit_df["pred_polarity"],
                                       average='macro')
        print(f"acc_explicit:{metrics_acc_explicit},f1_explicit:{metrics_f1_explicit}")

    ########################################
    #        w/o two training stages       #
    ########################################
    path = f'./testing_res_14/main_result_and_ablation_study_english/wo_all_stages_llama'
    os.makedirs(path, exist_ok=False)
    model_initialise = FinalPolarityAgentRaw(
                                             cot_questions_final=cot_raw,
                                             testing_datasets=dataset_output,
                                             prompt_path='./prompt_templates_res14/english_version'
                                                         '/res14_polarity_agent_eng_prompt.yaml')
    prompt = model_initialise.prompt_evaluate()
    result_df, metrics_test = model_initialise.evaluate_local()
    df_output = result_df
    df_output['prompt'] = prompt
    df_output.to_excel(os.path.join(path, f'wo_all_training_stages_result.xlsx'))
    print(metrics_test)
    with open(os.path.join(path, f'wo_all_training_stages_metrics.txt'), 'w') as file:
        file.write(str(metrics_test) + '\n')
        file.write(cot_raw)

    # calculate metrics
    df_result = pd.read_excel(os.path.join(path, f'wo_all_training_stages_result.xlsx'))
    df_result = df_result.rename(columns={'comment': 'sentence_text', 'aspect_name': 'aspect_term'})
    df_result = df_result.drop(columns='Unnamed: 0')
    df_result = df_result[['aspect_term', 'sentence_text', 'pred_polarity']]
    df_res_14_test = df_res_14_test.reset_index()
    df_res_14_test = df_res_14_test.drop(columns=['index'])
    combined_df = pd.merge(df_res_14_test, df_result, left_index=True, right_index=True, how='inner')
    implicit_df = combined_df[combined_df['implicit_sentiment'] == 'True']
    implicit_df['pred_polarity'].fillna("neutral", inplace=True)
    explicit_df = combined_df[combined_df['implicit_sentiment'] == 'False']
    explicit_df['pred_polarity'].fillna("neutral", inplace=True)
    metrics_acc_implicit = accuracy_score(y_true=implicit_df["polarity"], y_pred=implicit_df["pred_polarity"])
    metrics_f1_implicit = f1_score(y_true=implicit_df["polarity"], y_pred=implicit_df["pred_polarity"],
                                   average='macro')
    print(f"w/o two training stages")
    print(f"acc_implicit:{metrics_acc_implicit},f1_implicit:{metrics_f1_implicit}")
    metrics_acc_explicit = accuracy_score(y_true=explicit_df["polarity"], y_pred=explicit_df["pred_polarity"])
    metrics_f1_explicit = f1_score(y_true=explicit_df["polarity"], y_pred=explicit_df["pred_polarity"],
                                   average='macro')
    print(f"acc_explicit:{metrics_acc_explicit},f1_explicit:{metrics_f1_explicit}")

    ########################################
    #         w/o training stage II        #
    ########################################
    path = f'./testing_res_14/main_result_and_ablation_study_english/wo_stage_two_llama'
    os.makedirs(path, exist_ok=False)
    model_initialise = FinalPolarityAgentRaw(
                                             cot_questions_final=cot_final,
                                             testing_datasets=dataset_output,
                                             prompt_path='./prompt_templates_res14/english_version'
                                                         '/res14_polarity_agent_eng_prompt.yaml')
    prompt = model_initialise.prompt_evaluate()
    result_df, metrics_test = model_initialise.evaluate_local()
    df_output = result_df
    df_output['prompt'] = prompt
    df_output.to_excel(os.path.join(path, f'wo_stage_two_result.xlsx'))
    print(metrics_test)
    with open(os.path.join(path, f'wo_stage_two_metrics.txt'), 'w') as file:
        file.write(str(metrics_test) + '\n')
        file.write(cot_raw)

    # calculate metrics
    df_result = pd.read_excel(os.path.join(path, f'wo_stage_two_result.xlsx'))
    df_result = df_result.rename(columns={'comment': 'sentence_text', 'aspect_name': 'aspect_term'})
    df_result = df_result.drop(columns='Unnamed: 0')
    df_result = df_result[['aspect_term', 'sentence_text', 'pred_polarity']]
    df_res_14_test = df_res_14_test.reset_index()
    df_res_14_test = df_res_14_test.drop(columns=['index'])
    combined_df = pd.merge(df_res_14_test, df_result, left_index=True, right_index=True, how='inner')
    implicit_df = combined_df[combined_df['implicit_sentiment'] == 'True']
    implicit_df['pred_polarity'].fillna("neutral", inplace=True)
    explicit_df = combined_df[combined_df['implicit_sentiment'] == 'False']
    explicit_df['pred_polarity'].fillna("neutral", inplace=True)
    metrics_acc_implicit = accuracy_score(y_true=implicit_df["polarity"], y_pred=implicit_df["pred_polarity"])
    metrics_f1_implicit = f1_score(y_true=implicit_df["polarity"], y_pred=implicit_df["pred_polarity"],
                                   average='macro')
    print(f" w/o training stage II")
    print(f"acc_implicit:{metrics_acc_implicit},f1_implicit:{metrics_f1_implicit}")
    metrics_acc_explicit = accuracy_score(y_true=explicit_df["polarity"], y_pred=explicit_df["pred_polarity"])
    metrics_f1_explicit = f1_score(y_true=explicit_df["polarity"], y_pred=explicit_df["pred_polarity"],
                                   average='macro')
    print(f"acc_explicit:{metrics_acc_explicit},f1_explicit:{metrics_f1_explicit}")
