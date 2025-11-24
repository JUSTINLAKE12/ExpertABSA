import os
from dotenv import load_dotenv
from Agent_pools import ReasonExpert, PolarityAgentBench, GuidanceAgent
from KBS_constant import replace_double_braces, history_performances_handler_record_agent, guidance_handler, \
    extract_sample_dist, chat_model_local
import json
import pandas as pd

few_shot_reason_expert = [{
    "history_performances": {
        "current_round": 3,
        "key_insights": {
            "performance_classifications": "EVEN",
            "insight_summary": "Despite efforts to simplify the process and focus directly on sentiment polarity judgment in the third round of question design, the overall F1 score slightly decreased compared to the second round, while accuracy increased. This suggests that the adjustments in questions somewhat reduced misclassification but might have compromised the granularity in distinguishing positive and negative sentiments. Therefore, future question designs should continue to emphasize identifying direct sentiment words while effectively capturing indirect sentiment clues to optimize question structure for better balancing precision and recall, avoiding information loss due to over-simplification."
        }
    },
    "CoT_questions": "Q1: Does the review contain direct positive or negative sentiment words or strong sentiment clues about {{aspect_name}}? Q2: Even if the review does not directly mention {{aspect_name}}, are there any descriptions that clearly imply a positive or negative sentiment towards it? Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards {{aspect_name}} in the review?",
    "NegPos_error": {
        "number_of_mistakes": 4,
        "error_sample": {
            "comment": "Average battery life that's been hyped up a lot.",
            "aspect_name": "battery life",
            "polarity": "negative",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "\n-Q1: Does the review contain direct positive or negative sentiment words or strong sentiment clues about battery life?\n-Q2: Even if the review does not directly mention battery life, are there any descriptions that clearly imply a positive or negative sentiment towards it?\n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards battery life in the review?",
                    "CoT_sub_answers": "-A1: The review contains direct positive sentiment words or strong sentiment clues about battery life.\n-Q2: Even if the review does not directly mention battery life, there are descriptions that clearly imply a positive sentiment.\n-Q3: Based on the answers to the previous two questions, I think the sentiment polarity towards battery life in the review is positive."
                }
            ],
            "pred_polarity": "positive"
        }
    },
    "NegNeu_error": {
        "number_of_mistakes": 13,
        "error_sample": {
            "comment": "For the price you pay for this laptop, you'd expect it to be at least on par with other premium brands.",
            "aspect_name": "price",
            "polarity": "negative",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the review contain direct positive or negative sentiment words or strong sentiment clues about the price?\n-Q2: Even if the review does not directly mention the price, are there any descriptions that clearly imply a positive or negative sentiment towards it?\n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the price in the review?",
                    "CoT_sub_answers": "-A1: The review does not contain direct positive or negative sentiment words or strong sentiment clues about the price.\n-A2: The review mentions 'for the price you pay for this laptop,' which indicates some expectation about the price but does not clearly express a positive or negative sentiment.\n-A3: Based on the answers to the previous two questions, I think the sentiment polarity towards the price in the review is neutral."
                }
            ],
            "pred_polarity": "neutral"
        }
    },
    "PosNeg_error": {
        "number_of_mistakes": 9,
        "error_sample": {
            "comment": "The performance was decent, but not great.",
            "aspect_name": "performance",
            "polarity": "positive",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the review contain direct positive or negative sentiment words or strong sentiment clues about performance?\n-Q2: Even if the review does not directly mention performance, are there any descriptions that clearly imply a positive or negative sentiment towards it?\n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards performance in the review?",
                    "CoT_sub_answers": "-A1: The review contains direct negative sentiment words or strong sentiment clues about performance.\n-A2: The review does not provide any descriptions that clearly imply a positive or negative sentiment towards performance.\n-A3: Based on the answers to the previous two questions, I think the sentiment polarity towards performance in the review is negative."
                }
            ],
            "pred_polarity": "negative"
        }
    },
    "PosNeu_error": {
        "number_of_mistakes": 16,
        "error_sample": {
            "comment": "While there's a decent selection of ports, it shouldn't take this long to transfer files.",
            "aspect_name": "ports",
            "polarity": "positive",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the review contain direct positive or negative sentiment words or strong sentiment clues about the ports?\n-Q2: Even if the review does not directly mention the ports, are there any descriptions that clearly imply a positive or negative sentiment towards them?\n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the ports in the review?",
                    "CoT_sub_answers": "-A1: The review does not contain direct positive or negative sentiment words or strong sentiment clues about the ports.\n-A2: The review mentions the ports but does not clearly describe a positive or negative sentiment towards them.\n-A3: Based on the answers to the previous two questions, I think the sentiment polarity towards the ports in the review is neutral."
                }
            ],
            "pred_polarity": "neutral"
        }
    },
    "NeuPos_error": {
        "number_of_mistakes": 75,
        "error_sample": {
            "comment": "Try the new feature (not mentioned in the specs).",
            "aspect_name": "features",
            "polarity": "neutral",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the review contain direct positive or negative sentiment words or strong sentiment clues about the features?\n-Q2: Even if the review does not directly mention the features, are there any descriptions that clearly imply a positive or negative sentiment towards them?\n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the features in the review?",
                    "CoT_sub_answers": "-A1: The review does not contain direct positive or negative sentiment words or strong sentiment clues about the features.\n-A2: The review mentions 'new feature,' which might imply innovation.\n-A3: Based on the answers to the previous two questions, I think the sentiment polarity towards the features in the review is positive."
                }
            ],
            "pred_polarity": "positive"
        }
    },
    "NeuNeg_error": {
        "number_of_mistakes": 58,
        "error_sample": {
            "comment": "While there's a decent selection of ports, it shouldn't take this long to transfer files.",
            "aspect_name": "transfer speed",
            "polarity": "neutral",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the review contain direct positive or negative sentiment words or strong sentiment clues about the transfer speed?\n-Q2: Even if the review does not directly mention the transfer speed, are there any descriptions that clearly imply a positive or negative sentiment towards it?\n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the transfer speed in the review?",
                    "CoT_sub_answers": "-A1: The review contains negative sentiment words, such as 'too slow,' indicating slow transfer speed.\n-A2: The review does not provide other descriptions that clearly imply a positive or negative sentiment towards the transfer speed.\n-A3: Based on the answers to the previous two questions, the sentiment polarity towards the transfer speed in the review is negative."
                }
            ],
            "pred_polarity": "negative"
        }
    },
    "result": json.dumps({
        "logic_thinking_integrations": [
            {
                "logic_sub_questions": "Q1: Based on [the current round of error sets], among [negative, positive, neutral] sentiment polarities, which sentiment polarity judgment error is the most severe, analyze the reasons, and provide original examples.",
                "logic_sub_answers": "Among the provided error sets, the most severe misclassification is [neutral] sentiment polarity, especially misclassifying it as [positive]. In the provided error sample, the review 'Try the new feature (not mentioned in the specs)' mentions a special feature not included in the specs but does not directly express a positive sentiment; it was mistakenly interpreted as positive. This indicates that the current ABSA sub-question list may not sufficiently distinguish indirect mentions and actual sentiment polarity, particularly in neutral sentiment judgments. Additionally, in the [negative] sentiment error sample, the review 'For the price you pay for this laptop, you'd expect it to be at least on par with other premium brands' implicitly criticizes the high price but was misunderstood as neutral. This shows that the ABSA questions failed to sufficiently identify indirect sentiment clues."
            },
            {
                "logic_sub_questions": "Q2: Summarize the analysis of Q1 and, combined with the 'insight summary' of [ABSA sub-question list design key recommendations], provide suggestions for optimizing the [current round ABSA sub-question list] and explain the reasons.",
                "logic_sub_answers": "Based on Q1's analysis, the current ABSA sub-question list failed to [sufficiently distinguish indirect mentions and actual sentiment polarity, especially in identifying neutral sentiment] and [effectively identify indirect sentiment clues]. Optimization recommendations should focus on enhancing the precise identification of indirect sentiment clues, particularly in distinguishing neutral and positive sentiments. Additionally, more detailed assessment of the intensity of sentiment words and their contextual descriptions is needed to reduce misinterpretations caused by over-interpreting indirect information. Combining the emphasis on [identifying direct sentiment words] and [effectively capturing indirect sentiment clues], we should incorporate considerations of sentiment intensity and contextual nuances into the sub-questions to improve judgment accuracy."
            },
            {
                "logic_sub_questions": "Q3: Based on the optimization recommendations in Q2, choose tools from the toolbar and optimize the ABSA sub-question list. What is the optimized ABSA sub-question list?",
                "logic_sub_answers": "Using [logic_add] and [logic_modify] tools, the optimized ABSA sub-question list is as follows: Q1: Does the review directly mention {{aspect_name}} and use strong positive or negative sentiment words? Q2: Even if the review does not directly mention {{aspect_name}}, are there any descriptions that indirectly imply a positive or negative sentiment towards {{aspect_name}}? Q3: Does the review only provide a neutral description of {{aspect_name}}? Q4: Based on the answers to the previous three questions, what do you think is the overall sentiment polarity towards {{aspect_name}} in the review?"
            },
            {
                "logic_sub_questions": "Q4: Describe the differences between the optimized ABSA sub-question list and the [current round ABSA sub-question list].",
                "logic_sub_answers": "The optimized ABSA sub-question list uses [logic_add] to add Q3, specifically asking if there is only a neutral description. This helps more accurately distinguish neutral sentiment from other sentiment polarities, especially avoiding misinterpreting neutral descriptions as positive sentiment. Additionally, [logic_modify] adjusted Q1 and Q2 to focus more on the intensity of sentiment words and their context, reducing errors caused by over-simplification or misinterpretation of indirect sentiment clues."
            }
        ],
        "Tool_used": "logic_add,logic_modify",
        "optimised_absa_cot_questions": [
            {
                "question_index": 1,
                "optimised_absa_sub_questions": "Q1: Does the review directly mention {{aspect_name}} and use strong positive or negative sentiment words?"
            },
            {
                "question_index": 2,
                "optimised_absa_sub_questions": "Q2: Even if the review does not directly mention {{aspect_name}}, are there any descriptions that indirectly imply a positive or negative sentiment towards {{aspect_name}}?"
            },
            {
                "question_index": 3,
                "optimised_absa_sub_questions": "Q3: Does the review only provide a neutral description of {{aspect_name}}, without any clear positive or negative sentiment?"
            },
            {
                "question_index": 4,
                "optimised_absa_sub_questions": "Q4: Based on the answers to the previous three questions, what do you think is the overall sentiment polarity towards {{aspect_name}} in the review?"
            }
        ]
    }, ensure_ascii=False)
}]

few_shot_example_guidance = [
    {
        'history_performances': {
            "records": {
                "initial_performances": {
                    "initial_cot_questions": "Q1: What do you think is the sentiment polarity towards {aspect_name} in the computer review?\nQ2: Does the review contain any negative comments about {aspect_name}?\nQ3: Does the review contain any positive comments about {aspect_name}?\nQ4: Does the review contain any neutral comments about {aspect_name}?\nQ5: Based on the answers to Q2, Q3, and Q4, what do you think is the sentiment polarity towards {aspect_name} in the computer review?",
                    "initial_metrics": {'f1_score': 0.90, 'accuracy': 0.91}
                },
                "round_1_performances": {
                    'cot_questions_adjusted_based_on_initial_cot_questions': "Q1: What is your overall evaluation of the computer, including aspects such as food quality, service level, atmosphere, price reasonableness, and location convenience? Please elaborate on the sentiment for each aspect.\nQ2: In this review, did you find any negative comments about food quality, service level, atmosphere, price reasonableness, and location convenience? Please elaborate on the negative comments for each aspect.\nQ3: In this review, did you find any positive comments about food quality, service level, atmosphere, price reasonableness, and location convenience? Please elaborate on the positive comments for each aspect.\nQ4: In this review, did you find any neutral comments about food quality, service level, atmosphere, price reasonableness, and location convenience? Please elaborate on the neutral comments for each aspect.\nQ5: Based on the detailed answers to Q2, Q3, and Q4 about food quality, service level, atmosphere, price reasonableness, and location convenience, what do you think is the sentiment polarity towards {aspect_name} in this review?",
                    'round_1_metrics': {'f1_score': 0.7, 'accuracy': 0.69}
                }
            }
        },
        "result": json.dumps({
            "performance_classifications": 'WORSE',
            "insight_summary": "The second round of COT question design involved multiple dimensions (e.g., food quality, service level, atmosphere), deviating from the goal of predicting the sentiment polarity of a single {aspect_name}. This lack of focus and specificity led to confusion and misjudgment, reducing the accuracy and efficiency of sentiment polarity prediction. In contrast, the initial COT question design was more focused, with each question targeting sentiment information about {aspect_name}, thereby improving judgment accuracy and efficiency. Therefore, when designing ABSA sentiment polarity prediction sub-questions for computer reviews, the questions should be concise and focused on the sentiment information of a single {aspect_name}. This approach can more accurately capture and predict the sentiment polarity of different aspects in computer reviews, improving the overall prediction accuracy and reliability."
        }, ensure_ascii=False)
    },
    {
        'history_performances': {
            "records": {
                "round_2_performances": {
                    "cot_questions_adjusted_based_on_round_1_cot_questions": "What do you think is the sentiment polarity towards {aspect_name} in the computer review?",
                    "round_2_metrics": {
                        "f1_score": 0.91, "accuracy": 0.92
                    }
                },
                "round_3_performances": {
                    "cot_questions_adjusted_based_on_round_2_cot_questions": "Q1: What are the specific descriptions about {aspect_name} in the review?\nQ2: Do these descriptions imply a positive or negative sentiment towards {aspect_name}?\nQ3: Considering the overall context of the review, which sentiment polarity is more inclined towards {aspect_name}?",
                    "round_3_metrics": {
                        "f1_score": 0.93, "accuracy": 0.94
                    }
                }
            }
        },
        "result": json.dumps({
            "performance_classifications": 'BETTER',
            "insight_summary": "The second round of COT question design was overly simplified and lacked intermediate steps. The third round of COT question design was clear and specific, gradually guiding and considering the overall context, thereby improving the accuracy and reliability of sentiment polarity prediction. Therefore, in ABSA sentiment polarity prediction for computer reviews, the COT question design should ensure step-by-step guidance with clear questions and consider the overall context to improve prediction accuracy."
        }, ensure_ascii=False)
    }
]

with open('LAP14/lap_14_train.json', 'r') as file:
    lap14 = json.load(file)
dataset_output = []
for i in range(len(lap14)):
    dataset = {'comment': lap14[i]['sentence_text'],
               'aspect_name': lap14[i]['aspect_term'],
               'polarity': lap14[i]['polarity']}
    dataset_output.append(dataset)
experiment_dataset = extract_sample_dist(pd.DataFrame(dataset_output)).to_dict(orient='records')

if __name__ == '__main__':
    load_dotenv()
    epochs_env = int(os.getenv("EPOCHS"))
    for j in range(0, 3):
        records = {
            "records": {
            }
        }
        guidance = {}
        CoT_questions = """
        - Q1：whats the sentiment polarity of {aspect_name} in this laptop review?
        """
        for i in range(0, epochs_env):
            env_path = os.getenv("EXPERIMENT_PATH")
            path = f'{env_path}/lap_14_experiment_{i}'
            os.makedirs(path, exist_ok=False)
            ########################################
            #              polarity_agent          #
            ########################################
            # agent_predictions
            prompt_polarity = PolarityAgentBench(experiment_datasets=experiment_dataset,
                                                 cot_questions=CoT_questions,
                                                 path='prompt_templates_lap14/english_version/lap14_polarity_agent.yaml').prompt_evaluate()
            df_raw, metrics_raw = PolarityAgentBench(experiment_datasets=experiment_dataset,
                                                     cot_questions=CoT_questions,
                                                     path='prompt_templates_lap14/english_version/lap14_polarity_agent.yaml').evaluate_local(
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
                                               path='prompt_templates_lap14/english_version'
                                                    '/lap14_guidance_agent_eng_prompt.yaml').prompt_evaluate()
                with open(os.path.join(path, f'guidance_prompt_epochs_{i}.txt'), 'w') as file:
                    file.write(Insight_prompt)
                insight_advice = GuidanceAgent(chat_model=model_guidance,
                                               history_performances=history_performances,
                                               few_shot_example=few_shot_example_guidance,
                                               path='prompt_templates_lap14/english_version'
                                                    '/lap14_guidance_agent_eng_prompt.yaml').predict()
                guidance_handler(insights=guidance, history_advice=insight_advice, number_of_epochs=i)
                print(guidance)
            else:
                pass

            ########################################
            #          Reason_expert_Agent         #
            ########################################
            model_reason = chat_model_local(thinking_mode=True)
            cot_questions = CoT_questions.format(product_type="{{product_type}}", aspect_name="{{aspect_name}}")
            prompt = ReasonExpert(
                path='prompt_templates_lap14/english_version/lap14_reason_expert_eng_prompt.yaml',
                df_result=df_raw,
                metrics_result=metrics_raw,
                chat_model=model_reason,
                cot_questions=cot_questions, history_performances=guidance,
                few_shot_example=few_shot_reason_expert).prompt_evaluate()
            with open(os.path.join(path, f'reason_expert_prompt_epochs_{i}.txt'), 'w') as file:
                file.write(prompt)
            advice = ReasonExpert(df_result=df_raw, metrics_result=metrics_raw, chat_model=model_reason,
                                  cot_questions=cot_questions, history_performances=guidance,
                                  path='prompt_templates_lap14/english_version/lap14_reason_expert_eng_prompt.yaml',
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
