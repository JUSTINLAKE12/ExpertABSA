import os
import json
import pandas as pd
from dotenv import load_dotenv
from Agent_pools import ReasonExpert, PolarityAgent, GuidanceAgent
from KBS_constant import replace_double_braces, history_performances_handler_record_agent, \
    guidance_handler, \
    extract_sample_dist, chat_model_local

few_shot_example_guidance = [
    {
        "history_performances": {
            "records": {
                "round_0": {
                    "cot_questions": ["..."],  # list of minimal CoT steps
                    "metrics": {"f1_score": 0.0, "accuracy": 0.0}
                },
                "round_1": {
                    "cot_questions": ["..."],
                    "metrics": {"f1_score": 0.0, "accuracy": 0.0}
                }
            }
        },
        "result": {
            "performance_classifications": "<BETTER|WORSE|SAME>",
            "insight_summary": "..."  # one short sentence
        }
    }
]
few_shot_reason_expert = [{
    "history_performances": {
        "current_round": 0,
        "key_insights": {
            "performance_classifications": "<BETTER|WORSE|EVEN>",
            "insight_summary": "..."
        }
    },
    "CoT_questions": "Q1: ... Q2: ... Q3: ...",
    "NegPos_error": {
        "number_of_mistakes": 0,
        "error_sample": {
            "comment": "...",
            "aspect_name": "...",
            "polarity": "negative",
            "pred_CoT_reasons": [{"CoT_sub_questions": "...", "CoT_sub_answers": "..."}],
            "pred_polarity": "positive"
        }
    },
    "NegNeu_error": {
        "number_of_mistakes": 0,
        "error_sample": {
            "comment": "...",
            "aspect_name": "...",
            "polarity": "negative",
            "pred_CoT_reasons": [{"CoT_sub_questions": "...", "CoT_sub_answers": "..."}],
            "pred_polarity": "neutral"
        }
    },
    "PosNeg_error": {
        "number_of_mistakes": 0,
        "error_sample": {
            "comment": "...",
            "aspect_name": "...",
            "polarity": "positive",
            "pred_CoT_reasons": [{"CoT_sub_questions": "...", "CoT_sub_answers": "..."}],
            "pred_polarity": "negative"
        }
    },
    "PosNeu_error": {
        "number_of_mistakes": 0,
        "error_sample": {
            "comment": "...",
            "aspect_name": "...",
            "polarity": "positive",
            "pred_CoT_reasons": [{"CoT_sub_questions": "...", "CoT_sub_answers": "..."}],
            "pred_polarity": "neutral"
        }
    },
    "NeuPos_error": {
        "number_of_mistakes": 0,
        "error_sample": {
            "comment": "...",
            "aspect_name": "...",
            "polarity": "neutral",
            "pred_CoT_reasons": [{"CoT_sub_questions": "...", "CoT_sub_answers": "..."}],
            "pred_polarity": "positive"
        }
    },
    "NeuNeg_error": {
        "number_of_mistakes": 0,
        "error_sample": {
            "comment": "...",
            "aspect_name": "...",
            "polarity": "neutral",
            "pred_CoT_reasons": [{"CoT_sub_questions": "...", "CoT_sub_answers": "..."}],
            "pred_polarity": "negative"
        }
    },
    "result": {
        "logic_thinking_integrations": [
            {"logic_sub_questions": "Q1: ...", "logic_sub_answers": "..."},
            {"logic_sub_questions": "Q2: ...", "logic_sub_answers": "..."},
            {"logic_sub_questions": "Q3: ...", "logic_sub_answers": "..."},
            {"logic_sub_questions": "Q4: ...", "logic_sub_answers": "..."}
        ],
        "Tool_used": "logic_add,logic_modify",
        "optimised_absa_cot_questions": [
            {"question_index": 1, "optimised_absa_sub_questions": "Q1: ..."},
            {"question_index": 2, "optimised_absa_sub_questions": "Q2: ..."},
            {"question_index": 3, "optimised_absa_sub_questions": "Q3: ..."},
            {"question_index": 4, "optimised_absa_sub_questions": "Q4: ..."}
        ]
    }
}]

df = pd.read_json('commodity_dataset/commodity25_train.json')


def pick_first_nonempty(row):
    for col in ['positive_sentences', 'neutral_sentences', 'negative_sentences']:
        val = row[col]
        if val not in [None, [], '', 'nan']:  # treat None/empty as missing
            return val
    return None


df['comment'] = df.apply(pick_first_nonempty, axis=1)

dataset = {'comment': df['comment'],
           'product_type': df['product_type'],
           'aspect_name': df['aspect_name'],
           'polarity': df['sentiment']}

dataset = pd.DataFrame(dataset).to_dict(orient='records')

if __name__ == '__main__':
    load_dotenv()
    env_epochs = int(os.getenv("EPOCHS"))
    env_path = os.getenv("EXPERIMENT_PATH")
    for j in range(0, 3):
        records = {
            "records": {
            }
        }
        guidance = {}
        CoT_questions = """
        - Q1：whats the sentiment polarity of {aspect_name} in this news?
        """
        for i in range(0, env_epochs):
            path = f'{env_path}/comm_experiment_{i}'
            os.makedirs(path, exist_ok=False)
            ########################################
            #            polarity_agent            #
            ########################################
            # agent_predictions
            prompt_polarity = PolarityAgent(experiment_datasets=dataset,
                                            cot_questions=CoT_questions,
                                            path='./prompt_templates_comm25/commodity_polarity_agent.yaml').prompt_evaluate()
            df_raw, metrics_raw = PolarityAgent(experiment_datasets=dataset,
                                                cot_questions=CoT_questions,
                                                path='./prompt_templates_comm25/commodity_polarity_agent.yaml').evaluate_local(
                thinking_mode=False)
            df_raw.to_excel(os.path.join(path, f'polarity_epochs{i}.xlsx'))
            print(metrics_raw)
            with open(os.path.join(path, f'polarity_result_epoch_{i}.txt'), 'w') as file:
                file.write(str(metrics_raw) + '\n')
                file.write(str(CoT_questions))
            # history_recording
            history_performances_handler_record_agent(records=records, metrics=metrics_raw, cot_questions=CoT_questions,
                                                      number_of_epochs=i)
            history_performances = json.dumps(records, ensure_ascii=False, indent=4)

            ########################################
            #             guidance_Agent           #
            ########################################
            if i > 0:
                model_guidance = chat_model_local(thinking_mode=True)
                Insight_prompt = GuidanceAgent(chat_model=model_guidance,
                                               history_performances=history_performances,
                                               few_shot_example=few_shot_example_guidance,
                                               path='./prompt_templates_comm25/commodity_guidance_agent.yaml').prompt_evaluate()
                with open(os.path.join(path, f'guidance_prompt_epochs_{i}.txt'), 'w') as file:
                    file.write(Insight_prompt)
                insight_advice = GuidanceAgent(chat_model=model_guidance,
                                               history_performances=history_performances,
                                               few_shot_example=few_shot_example_guidance,
                                               path='./prompt_templates_comm25/commodity_guidance_agent.yaml').predict()
                guidance_handler(insights=guidance, history_advice=insight_advice, number_of_epochs=i)
                print(guidance)
            else:
                pass

            ########################################
            #          Reason_expert_Agent         #
            ########################################
            model_reason = chat_model_local(thinking_mode=True)
            cot_questions = CoT_questions.format(aspect_name="{{aspect_name}}")
            prompt = ReasonExpert(
                path='./prompt_templates_comm25/commodity_reason_expert_agent.yaml',
                df_result=df_raw,
                metrics_result=metrics_raw,
                chat_model=model_reason,
                cot_questions=cot_questions, history_performances=guidance,
                few_shot_example=few_shot_reason_expert).prompt_evaluate()
            with open(os.path.join(path, f'reason_expert_prompt_epochs_{i}.txt'), 'w') as file:
                file.write(prompt)
            advice = ReasonExpert(df_result=df_raw, metrics_result=metrics_raw, chat_model=model_reason,
                                  cot_questions=cot_questions, history_performances=guidance,
                                  path='./prompt_templates_comm25/commodity_reason_expert_agent.yaml',
                                  few_shot_example=few_shot_reason_expert).predict()
            with open(os.path.join(path, f'reason_expert_advice_epochs_{i}.txt'), 'w') as file:
                file.write(str(advice))

            ########################################
            #             data_handling            #
            ########################################
            questions = advice['optimised_absa_cot_questions']
            CoT_questions_raw = []
            for i in range(len(questions)):
                CoT_questions_raw.append("- " + questions[i]['optimised_absa_sub_questions'])
            CoT_questions_intermediate = '\n'.join(CoT_questions_raw)
            CoT_questions = replace_double_braces(
                CoT_questions_intermediate)
            print(CoT_questions)
