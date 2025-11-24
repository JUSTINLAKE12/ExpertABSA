import json
import pandas as pd
from few_shots_example_agent import ErrorCorrectionAgent

if __name__ == '__main__':
    ########################################
    # Training Stage I result collections  #
    ########################################
    cot_final = """
        {input your desired optimised chain of thoughts questions list here}
    """

    result_df = pd.read_excel("{input the corresponding error dataset path of cot_final}")

    # # for better performance of ErrorCorrectionAgent you should at least hand-craft one example as few-shots
    # example to error correction agent
    hand_crafted_examples = ['{input your hand-crafted example here in dict format}']

    error_result_df = result_df[result_df['polarity'] != result_df['pred_polarity']]
    error_df = error_result_df.drop(columns=['pred_CoT_reasons', 'pred_polarity', 'error'])
    error_df = error_df.sort_index(ignore_index=True)
    error_df_final = error_df.to_dict(orient='records')

    ########################################
    #        Error_correction Agent        #
    ########################################
    # before running error correction agent, choose correct pre_defined prompt template in prompt_template files
    model_initialise = ErrorCorrectionAgent(path='{prompt_template_path}',
                                            cot_questions_final=cot_final, cot_questions_result_df=error_df_final,
                                            few_shot_example=hand_crafted_examples)
    # for english prompt
    df = model_initialise.predict_local(thinking_mode=False)

    ########################################
    #           Result_Gathering           #
    ########################################
    error_df_combine = error_df[['comment', 'aspect_name', 'polarity']]
    df_result = pd.DataFrame(df)
    df_complete = df_result.join(error_df_combine)
    df_complete = df_complete[df_complete['pred_polarity'] == df_complete['polarity']]
    df_dict = df_complete.to_dict(orient='records')

    few_shots_pools = []
    for i in range(len(df_dict)):
        static_example = {
            "comment": df_dict[i]["comment"],
            "aspect_name": df_dict[i]["aspect_name"],
            "result": json.dumps({
                "cot_reasons": df_dict[i]["cot_reasons"],
                "pred_polarity": df_dict[i]["pred_polarity"]
            }, ensure_ascii=False)
        }
        few_shots_pools.append(static_example)

    with open('./correction_agent_output/few_shots_pools.json', 'w', encoding='UTF-8') as file:
        json.dump(few_shots_pools, file, indent=4, ensure_ascii=False)
