import json
import re
import os
from dotenv import load_dotenv
from urllib.parse import urljoin

import pandas as pd
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI


def positive_error_processing(df: pd.DataFrame):
    error_dataset = df[df['polarity'] != df['pred_polarity']]
    columns_to_drop = ['doc_id', 'sub_id', 'init_example', 'error', 'reason']
    error_dataset_columns = [col for col in columns_to_drop if col in error_dataset.columns]
    error_dataset = error_dataset.drop(columns=error_dataset_columns)

    # positive_negative error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'positive') & (error_dataset['pred_polarity'] == 'negative')].empty == False:
        pos_neg_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'positive') & (error_dataset['pred_polarity'] == 'negative')]
        pos_neg_final = pos_neg_dataset.to_dict(orient='records')[0]
        pos_neg_final = {'number_of_mistakes': len(pos_neg_dataset),
                         'error_sample': pos_neg_final}

    else:
        pos_neg_final = '没有错误'

    # positive_neutral_error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'positive') & (error_dataset['pred_polarity'] == 'neutral')].empty == False:
        pos_neu_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'positive') & (error_dataset['pred_polarity'] == 'neutral')]
        pos_neu_final = pos_neu_dataset.to_dict(orient='records')[0]
        pos_neu_final = {'number_of_mistakes': len(pos_neu_dataset),
                         'error_sample': pos_neu_final}
    else:
        pos_neu_final = '没有错误'

    # positive_na_error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'positive') & (error_dataset['pred_polarity'] == 'N/A')].empty == False:
        pos_NA_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'positive') & (error_dataset['pred_polarity'] == 'N/A')]
        pos_na_final = pos_NA_dataset.to_dict(orient='records')[0]
        pos_na_final = {'number_of_mistakes': len(pos_NA_dataset),
                        'error_sample': pos_na_final}
    else:
        pos_na_final = '没有错误'
    return pos_neg_final, pos_neu_final, pos_na_final


def negative_error_processing(df: pd.DataFrame):
    error_dataset = df[df['polarity'] != df['pred_polarity']]
    columns_to_drop = ['doc_id', 'sub_id', 'init_example', 'error', 'reason']
    error_dataset_columns = [col for col in columns_to_drop if col in error_dataset.columns]
    error_dataset = error_dataset.drop(columns=error_dataset_columns)
    # negative_positive error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'negative') & (error_dataset['pred_polarity'] == 'positive')].empty == False:
        neg_pos_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'negative') & (error_dataset['pred_polarity'] == 'positive')]
        neg_pos_final = neg_pos_dataset.to_dict(orient='records')[0]
        neg_pos_final = {'number_of_mistakes': len(neg_pos_dataset),
                         'error_sample': neg_pos_final}
    else:
        neg_pos_final = '没有错误'

    # negative_neutral error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'negative') & (error_dataset['pred_polarity'] == 'neutral')].empty == False:
        neg_neu_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'negative') & (error_dataset['pred_polarity'] == 'neutral')]
        neg_neu_final = neg_neu_dataset.to_dict(orient='records')[0]
        neg_neu_final = {'number_of_mistakes': len(neg_neu_dataset),
                         'error_sample': neg_neu_final}

    else:
        neg_neu_final = '没有错误'

    # negative_na error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'negative') & (error_dataset['pred_polarity'] == 'N/A')].empty == False:
        neg_na_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'negative') & (error_dataset['pred_polarity'] == 'N/A')]
        neg_na_final = neg_na_dataset.to_dict(orient='records')[0]
        neg_na_final = {'number_of_mistakes': len(neg_na_dataset),
                        'error_sample': neg_na_final
                        }
    else:
        neg_na_final = '没有错误'
    return neg_na_final, neg_pos_final, neg_neu_final


def neutral_error_processing(df: pd.DataFrame):
    error_dataset = df[df['polarity'] != df['pred_polarity']]
    columns_to_drop = ['doc_id', 'sub_id', 'init_example', 'error', 'reason']
    error_dataset_columns = [col for col in columns_to_drop if col in error_dataset.columns]
    error_dataset = error_dataset.drop(columns=error_dataset_columns)
    # neutral_positive error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'neutral') & (error_dataset['pred_polarity'] == 'positive')].empty == False:
        neu_pos_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'neutral') & (error_dataset['pred_polarity'] == 'positive')]
        neu_pos_final = neu_pos_dataset.to_dict(orient='records')[0]
        neu_pos_final = {'number_of_mistakes': len(neu_pos_dataset),
                         'error_sample': neu_pos_final}

    else:
        neu_pos_final = '没有错误'

    # neutral_negative error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'neutral') & (error_dataset['pred_polarity'] == 'negative')].empty == False:
        neu_neg_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'neutral') & (error_dataset['pred_polarity'] == 'negative')]
        neu_neg_final = neu_neg_dataset.to_dict(orient='records')[0]
        neu_neg_final = {'number_of_mistakes': len(neu_neg_dataset),
                         'error_sample': neu_neg_final}
    else:
        neu_neg_final = '没有错误'

    # neutral_na error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'neutral') & (error_dataset['pred_polarity'] == 'N/A')].empty == False:
        neu_na_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'neutral') & (error_dataset['pred_polarity'] == 'N/A')]
        neu_na_final = neu_na_dataset.to_dict(orient='records')[0]
        neu_na_final = {'number_of_mistakes': len(neu_na_dataset),
                        'error_sample': neu_na_final}

    else:
        neu_na_final = '没有错误'

    return neu_na_final, neu_pos_final, neu_neg_final


def na_error_processing(df: pd.DataFrame):
    error_dataset = df[df['polarity'] != df['pred_polarity']]
    columns_to_drop = ['doc_id', 'sub_id', 'init_example', 'error', 'reason']
    error_dataset_columns = [col for col in columns_to_drop if col in error_dataset.columns]
    error_dataset = error_dataset.drop(columns=error_dataset_columns)
    # N/A to positive error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'N/A') & (error_dataset['pred_polarity'] == 'positive')].empty == False:
        na_pos_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'N/A') & (error_dataset['pred_polarity'] == 'positive')]
        na_pos_final = na_pos_dataset.to_dict(orient='records')[0]
        na_pos_final = {'number_of_mistakes': len(na_pos_dataset),
                        'error_sample': na_pos_final
                        }
    else:
        na_pos_final = '没有错误'

    # N/A to negative error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'N/A') & (error_dataset['pred_polarity'] == 'negative')].empty == False:
        na_neg_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'N/A') & (error_dataset['pred_polarity'] == 'negative')]
        na_neg_final = na_neg_dataset.to_dict(orient='records')[0]
        na_neg_final = {'number_of_mistakes': len(na_neg_dataset),
                        'error_sample': na_neg_final
                        }
    else:
        na_neg_final = '没有错误'

    # N/A to neutral error
    if error_dataset.loc[
        (error_dataset['polarity'] == 'N/A') & (error_dataset['pred_polarity'] == 'neutral')].empty == False:
        na_neu_dataset = error_dataset.loc[
            (error_dataset['polarity'] == 'N/A') & (error_dataset['pred_polarity'] == 'neutral')]
        na_neu_final = na_neu_dataset.to_dict(orient='records')[0]
        na_neu_final = {'number_of_mistakes': len(na_neu_dataset),
                        'error_sample': na_neu_final
                        }
    else:
        na_neu_final = '没有错误'

    return na_pos_final, na_neg_final, na_neu_final


def catch_when_invoke(chain_obj: Runnable, _input: dict):
    try:
        result = chain_obj.invoke(_input)
        if result is None or not isinstance(result, dict):
            return {"error": "LLM返回空值"}
        return result
    except Exception as e:
        return {"error": str(e)}


def strip_whitespaces(text: str) -> str:
    from string import whitespace
    return text.strip(whitespace)


def history_performances_handler_record_agent(records, metrics: str, cot_questions: str, number_of_epochs: int):
    if len(records['records']) <= 2:
        if number_of_epochs == 0:
            records['records'].update(
                {f"Initial_performances": {
                    'initial_cot_questions': cot_questions,
                    'initial_metrics': metrics}})
            return json.dumps(records, indent=4, ensure_ascii=False)
        else:
            records['records'].update(
                {f"round_{number_of_epochs}_performances": {
                    f"cot_questions_adjusted_based_on_round_{number_of_epochs}": cot_questions,
                    f'round_{number_of_epochs}_metrics': metrics}})
            return json.dumps(records, indent=4, ensure_ascii=False)
    else:
        key_name = next(iter(records['records']))
        records['records'].pop(key_name)
        records['records'].update(
            {f"round_{number_of_epochs}_performances": {
                f'cot_questions_adjusted_based_on_round_{number_of_epochs}': cot_questions,
                f'round_{number_of_epochs}_metrics': metrics}})
        return json.dumps(records, indent=4, ensure_ascii=False)


def guidance_handler(insights, history_advice, number_of_epochs):
    # if len(insights) <= 2:
    insights.update({"current_round": number_of_epochs,
                     "key_insights": history_advice})
    # else:
    #     key_name = next(iter(insights))
    #     insights.pop(key_name)
    #     insights.update({round_{number_of_epochs}': number_of_epochs,
    #                      f'insights_round_{number_of_epochs}': history_advice})


# FIXME:
def chat_model_local(temperature: float = 0.1, thinking_mode: bool = False):
    load_dotenv()
    model_name = os.getenv("MODEL_NAME")
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")
    chat_model = ChatOpenAI(model=model_name,
                            base_url=base_url,
                            temperature=temperature,
                            openai_api_key=api_key,
                            max_tokens=5000,
                            model_kwargs={'extra_body': {'chat_template_kwargs':
                                                             {'enable_thinking': thinking_mode
                                                              }
                                                         }
                                          }
                            )
    return chat_model

def replace_double_braces(text):
    text = re.sub(r'\{\{', '{', text)
    text = re.sub(r'\}\}', '}', text)
    return text


def sample_items(group, n):
    return group.sample(n, random_state=1)


def extract_sample_dist(df):
    df = pd.DataFrame(df)

    # Total number of samples to extract
    total_samples = 500

    # Calculate the percentage distribution of each category
    category_counts = df['polarity'].value_counts()
    total_count = category_counts.sum()
    category_percentage = category_counts / total_count

    # Determine the number of samples to extract from each category based on the percentage distribution
    samples_per_category = (category_percentage * total_samples).round().astype(int)

    # Ensure the total number of samples equals the desired number by adjusting the largest category
    difference = total_samples - samples_per_category.sum()
    if difference != 0:
        largest_category = samples_per_category.idxmax()
        samples_per_category[largest_category] += difference
    sampled_dfs = [sample_items(df[df['polarity'] == category], n_samples) for category, n_samples in
                   samples_per_category.items()]

    # Combine the sampled items into a new DataFrame
    balanced_df = pd.concat(sampled_dfs).reset_index(drop=True)
    return balanced_df
