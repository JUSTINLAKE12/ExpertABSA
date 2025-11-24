import os
import json
import pandas as pd
from dotenv import load_dotenv
from Agent_pools import ReasonExpert, PolarityAgentBench, GuidanceAgent
from KBS_constant import replace_double_braces, history_performances_handler_record_agent, guidance_handler, \
    extract_sample_dist, chat_model_local


few_shot_reason_expert = [{
    "history_performances": {
        "current_round": 3,
        "key_insights": {
            "performance_classifications": "EVEN",
            "insight_summary": "Despite attempts to simplify the process and focus directly on sentiment polarity judgment in the third round of COT question design, its macro F1 score slightly decreased compared to the second round, while accuracy improved. This indicates that the adjustments in the questions have somewhat reduced misclassifications but may have compromised the granularity in distinguishing positive and negative sentiments. Therefore, future sub-question designs should continue to emphasize the identification of direct sentiment vocabulary while maintaining effective capture of indirect sentiment cues, and optimize question structures to better balance precision and recall, avoiding information loss due to oversimplification."
        }
    },
    "CoT_questions": "Does the comment contain direct positive or negative sentiment words or strong sentiment cues about {{aspect_name}}? Q2: Even if the comment does not directly mention {{aspect_name}}, is there any description that can clearly infer positive or negative sentiment about it? Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards {{aspect_name}} in the restaurant review?",
    "NegPos_error": {
        "number_of_mistakes": 4,
        "error_sample": {
            "comment": "Average cake thats been courted by a LOT of hype.",
            "aspect_name": "cake",
            "polarity": "negative",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "\n-Q1: Does the comment contain direct positive or negative sentiment words or strong sentiment cues about the cake?\n-Q2:Even if the comment does not directly mention the cake, is there any description that can clearly infer positive or negative sentiment about it? \n-Q3:Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the cake in the restaurant review?",
                    "CoT_sub_answers": "-A1: The comment contains direct positive sentiment words or strong sentiment cues about the cake.\n-Q2: Even if the comment does not directly mention the cake, there is a description that can clearly infer positive sentiment about it.\n-Q3: Based on the answers to the previous two questions, I think the sentiment polarity towards the cake in the restaurant review is positive."
                }
            ],
            "pred_polarity": "positive"
        }
    },
    "NegNeu_error": {
        "number_of_mistakes": 13,
        "error_sample": {
            "comment": "For the price you pay for the food here, you'd expect it to be at least on par with other Japanese restaurants.",
            "aspect_name": "price",
            "polarity": "negative",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the comment contain direct positive or negative sentiment words or strong sentiment cues about the price? \n-Q2: Even if the comment does not directly mention the price, is there any description that can clearly infer positive or negative sentiment about it? \n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the price in the restaurant review?",
                    "CoT_sub_answers": "-A1: The comment does not contain direct positive or negative sentiment words or strong sentiment cues about the price. \n-A2: The comment mentions 'For the price you pay for the food here,' which indicates that the reviewer has certain expectations for the price, but it does not explicitly express positive or negative sentiment about the price. \n-A3: Based on the answers to the previous two questions, I think the sentiment polarity towards the price in the restaurant review is neutral."
                }
            ],
            "pred_polarity": "neutral"
        }
    },
    "PosNeg_error": {
        "number_of_mistakes": 9,
        "error_sample": {
            "comment": "Food was decent, but not great.",
            "aspect_name": "Food",
            "polarity": "positive",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the comment contain direct positive or negative sentiment words or strong sentiment cues about the food? \n-Q2: Even if the comment does not directly mention the food, is there any description that can clearly infer positive or negative sentiment about it? \n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the food in the restaurant review?",
                    "CoT_sub_answers": "-A1: The comment contains direct negative sentiment words or strong sentiment cues about the food. \n-A2: The comment does not provide any description that can clearly infer positive or negative sentiment about the food. \n-A3: Based on the answers to the previous two questions, I think the sentiment polarity towards the food in the restaurant review is negative."
                }
            ],
            "pred_polarity": "negative"
        }
    },
    "PosNeu_error": {
        "number_of_mistakes": 16,
        "error_sample": {
            "comment": "While there's a decent menu, it shouldn't take ten minutes to get your drinks and 45 for a dessert pizza.",
            "aspect_name": "menu",
            "polarity": "positive",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the comment contain direct positive or negative sentiment words or strong sentiment cues about the menu? \n-Q2: Even if the comment does not directly mention the menu, is there any description that can clearly infer positive or negative sentiment about it? \n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the menu in the restaurant review?",
                    "CoT_sub_answers": "-A1:The comment does not contain direct positive or negative sentiment words or strong sentiment cues about the menu. \n-A2: The comment mentions the menu, but does not explicitly describe positive or negative sentiment about the menu. \n-A3: Based on the answers to the previous two questions, I think the sentiment polarity towards the menu in the restaurant review is neutral."
                }
            ],
            "pred_polarity": "neutral"
        }
    },
    "NeuPos_error": {
        "number_of_mistakes": 75,
        "error_sample": {
            "comment": "Try the rose roll (not on menu).",
            "aspect_name": "menu",
            "polarity": "neutral",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the comment contain direct positive or negative sentiment words or strong sentiment cues about the menu? \n-Q2: Even if the comment does not directly mention the menu, is there any description that can clearly infer positive or negative sentiment about it? \n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the menu in the restaurant review?",
                    "CoT_sub_answers": "-A1: The comment does not contain direct positive or negative sentiment words or strong sentiment cues about the menu. \n-A2: The comment mentions 'rose roll,' which is a special dish and may imply that the menu is diverse and rich. \n-A3: Based on the answers to the previous two questions, I think the sentiment polarity towards the menu in the restaurant review is positive."
                }
            ],
            "pred_polarity": "positive"
        }
    },
    "NeuNeg_error": {
        "number_of_mistakes": 58,
        "error_sample": {
            "comment": "While there's a decent menu, it shouldn't take ten minutes to get your drinks and 45 for a dessert pizza.",
            "aspect_name": "drinks",
            "polarity": "neutral",
            "pred_CoT_reasons": [
                {
                    "CoT_sub_questions": "-Q1: Does the comment contain direct positive or negative sentiment words or strong sentiment cues about the drinks? \n-Q2: Even if the comment does not directly mention the drinks, is there any description that can clearly infer positive or negative sentiment about them? \n-Q3: Based on the answers to the previous two questions, what do you think is the sentiment polarity towards the drinks in the restaurant review?",
                    "CoT_sub_answers": "-A1: The comment contains negative sentiment words, such as 'ten minutes' and '45 for a dessert pizza,' which suggest that the waiting time for the drinks is too long. \n-A2: There are no other descriptions in the comment that clearly infer positive or negative sentiment about the drinks. \n-A3: Based on the answers to the previous two questions, the sentiment polarity towards the drinks in the restaurant review is negative."
                }
            ],
            "pred_polarity": "negative"
        }
    },
    "result": json.dumps({
        "logic_thinking_integrations": [
            {
                "logic_sub_questions": "Q1:  Based on the [Current Round Error Set], under the [negative, positive, neutral] sentiment polarities, first determine which sentiment polarity judgment error is the most severe, analyze the reasons, and provide original examples.",
                "logic_sub_answers": "In the provided error set, the number of mistakes in classifying [neutral] sentiment polarity is the highest, particularly errors where it was incorrectly classified as [positive]. In the provided error sample, the comment 'Try the rose roll (not on menu)' mentions a special dish not on the menu, but this does not directly express positive sentiment about the menu; it was mistakenly interpreted as positive. This suggests that the current ABSA sub-question list may not adequately distinguish between indirect mentions and actual sentiment polarity, especially in neutral sentiment judgments. Additionally, in the example of errors with [negative] sentiment, the comment 'you'd expect it to be at least on par with other Japanese restaurants' implicitly criticizes the price for being too high but was misunderstood as neutral. This indicates that the ABSA questions failed to adequately identify indirect sentiment cues."
            },
            {
                "logic_sub_questions": "Q2: Summarize the analysis from Q1 and, by combining the [ABSA sub-questions List Design key suggestions] 'insight_summary', provide suggestions for optimizing the [Current Round ABSA Sub-questions List] and explain the reasons.",
                "logic_sub_answers": "Based on the analysis from Q1, the current ABSA sub-question list fails to [adequately distinguish between indirect mentions and actual sentiment polarity, especially in identifying neutral sentiments] and [effectively recognize indirect sentiment cues]. Optimization suggestions should focus on enhancing the precise identification of indirect sentiment cues, particularly in distinguishing between neutral and positive sentiments. Additionally, there needs to be a more detailed evaluation of the intensity of sentiment words and the context in which they are described to reduce misinterpretations caused by over-reading indirect information. Combining the design points that emphasize [recognition of direct sentiment words] and [effective capture of indirect sentiment cues], we should incorporate considerations of sentiment intensity and contextual nuances into the sub-questions to improve the accuracy of judgments."
            },
            {
                "logic_sub_questions": "Q3: Based on the optimization suggestions from Q2, select tools from the toolbar and optimize the ABSA sub-questions list. What is the optimized ABSA sub-questions list?",
                "logic_sub_answers": "Using the [logic_add] and [logic_modify] tools, the optimized ABSA sub-question list is as follows,Q1: Does the comment directly mention {{aspect_name}} and use strong positive or negative sentiment words?Q2: Even if the comment does not directly mention {{aspect_name}}, is there a description that can indirectly infer positive or negative sentiment about {{aspect_name}}?Q3: Does the comment only provide a neutral description of {{aspect_name}}?Q4: Based on the answers to the previous three questions, what do you think is the final sentiment polarity towards {{aspect_name}} in the restaurant review?"
            },
            {
                "logic_sub_questions": "Q4: Describe the differences between the optimized ABSA sub-questions list and the [Current Round ABSA Sub-questions List].",
                "logic_sub_answers": "The optimized ABSA sub-questions list uses [logic_add] to add Q3, which specifically asks whether there are only neutral descriptions. This helps to more accurately distinguish neutral sentiments from other sentiment polarities, particularly avoiding misinterpreting neutral descriptions as positive sentiment. Additionally, [logic_modify] is used to adjust Q1 and Q2 to focus more on the intensity of sentiment words and their context, reducing errors caused by oversimplification or misinterpretation of indirect sentiment cues."
            }
        ],
        "Tool_used": "logic_add,logic_modify",
        "optimised_absa_cot_questions": [
            {
                "question_index": 1,
                "optimised_absa_sub_questions": "Q1: Does the comment directly mention {{aspect_name}} and use strong positive or negative sentiment words?"
            },
            {
                "question_index": 2,
                "optimised_absa_sub_questions": "Q2: Even if the comment does not directly mention {{aspect_name}}, is there a description that can indirectly infer positive or negative sentiment about {{aspect_name}}?"
            },
            {
                "question_index": 3,
                "optimised_absa_sub_questions": "Q3: Does the comment only provide a neutral description of {{aspect_name}}, without any clear positive or negative sentiment?"
            },
            {
                "question_index": 4,
                "optimised_absa_sub_questions": "Q4: Based on the answers to the previous three questions, what do you think is the overall sentiment polarity towards {{aspect_name}} in the restaurant review?"
            }
        ]
    }, ensure_ascii=False)
}
]

few_shot_example_guidance = [
    {
        'history_performances': {
            "records": {
                "initial_performances": {
                    "initial_cot_questions": "Q1: What do you think is the sentiment polarity of {aspect_name} in the restaurant review?\nQ2: In the comment, is there any negative evaluation of {aspect_name}?\nQ3: In the comment, is there any positive evaluation of {aspect_name}?\nQ4: In the comment, is there any neutral evaluation of {aspect_name}?\nQ5: Based on the answers to Q2, Q3, and Q4, what do you think is the sentiment polarity of {aspect_name} in the restaurant review?",
                    "initial_metrics": {'f1_score': 0.90,
                                        'accuracy': 0.91}},
                "round_1_performances": {
                    'cot_questions_adjusted_based_on_initial_cot_questions': "Q1: What is your overall evaluation of the restaurant in terms of food quality, service level, atmosphere, price reasonableness, and location convenience? Please detail the sentiment for each aspect.\nQ2: In this review, did you find any negative evaluation of food quality, service level, atmosphere, price reasonableness, and location convenience? Please detail the negative evaluation for each aspect.\nQ3: In this review, did you find any positive evaluation of food quality, service level, atmosphere, price reasonableness, and location convenience? Please detail the positive evaluation for each aspect.\nQ4: In this review, did you find any neutral evaluation of food quality, service level, atmosphere, price reasonableness, and location convenience? Please detail the neutral evaluation for each aspect.\nQ5: Based on the detailed answers to Q2, Q3, and Q4 regarding food quality, service level, atmosphere, price reasonableness, and location convenience, what do you think is the sentiment polarity of {aspect_name} in this review?",
                    'round_1_metrics': {'f1_score': 0.7, 'accuracy': 0.69}}
            }},
        "result": json.dumps({
            "performance_classifications": 'WORSE',
            "insight_summary": 'The second round of COT question design involved multiple dimensions (such as food quality, service level, atmosphere, etc.), deviating from the goal of predicting the sentiment polarity of a single {aspect_name}. This lack of focus and specificity led to confusion and misjudgment, reducing the accuracy and efficiency of sentiment polarity prediction. In contrast, the initial COT question design was more focused, with each question targeting the sentiment information of {aspect_name}, thereby improving judgment accuracy and efficiency. Therefore, when designing ABSA sentiment polarity prediction sub-questions for restaurant reviews, the questions should be concise and clear, focusing on the sentiment information of a single {aspect_name}. This approach can more accurately capture and predict the sentiment polarity of different aspects in restaurant reviews, enhancing overall prediction accuracy and reliability.'
        }, ensure_ascii=False)},
    {
        'history_performances': {
            "records": {
                "round_2_performances": {
                    "cot_questions_adjusted_based_on_round_1_cot_questions": "What do you think is the sentiment polarity of {aspect_name} in the restaurant review?",
                    "round_2_metrics": {
                        "f1_score": 0.91,
                        "accuracy": 0.92
                    }
                },
                "round_3_performances": {
                    "cot_questions_adjusted_based_on_round_2_cot_questions": "Q1: What is the specific description of {aspect_name} in the comment?\nQ2: Do these descriptions imply a positive or negative evaluation of {aspect_name}?\nQ3: Considering the overall context of the comment, what is the sentiment polarity of {aspect_name} more inclined towards?",
                    "round_3_metrics": {
                        "f1_score": 0.93,
                        "accuracy": 0.94
                    }
                }
            }
        },
        "result": json.dumps({
            "performance_classifications": 'BETTER',
            "insight_summary": 'The second round of COT question design was too simplistic, lacking intermediate steps. The third round of COT question design was clear and specific, guiding step-by-step and considering the overall context, thereby improving the accuracy and reliability of sentiment polarity prediction. Therefore, in ABSA sentiment polarity prediction for restaurant reviews, COT question design should ensure indirectness while using clear guiding questions and considering the overall context to improve prediction accuracy.'
        }, ensure_ascii=False)
    }
]

records = {
    "records": {
    }
}
guidance = {}

with open('RES14/res_14_train.json', 'r') as file:
    res14 = json.load(file)
dataset_output = []
for i in range(len(res14)):
    dataset = {'comment': res14[i]['sentence_text'],
               'aspect_name': res14[i]['aspect_term'],
               'polarity': res14[i]['polarity']}
    dataset_output.append(dataset)
experiment_dataset = extract_sample_dist(pd.DataFrame(dataset_output)).to_dict(orient='records')

if __name__ == '__main__':
    load_dotenv()
    env_epochs = int(os.getenv("EPOCHS"))
    env_path = os.getenv("EXPERIMENT_PATH")
    for j in range(0, 3):
        CoT_questions = """
        - Q1：whats the sentiment polarity of {aspect_name} in this restaurant review?
        """
        for i in range(env_epochs):
            path = f'{env_path}/res14_experiment_{i}'
            os.makedirs(path, exist_ok=False)
            ########################################
            #             polarity_agent           #
            ########################################
            # agent_predictions
            prompt_polarity = PolarityAgentBench(experiment_datasets=experiment_dataset,
                                            cot_questions=CoT_questions,
                                            path='prompt_templates_res14/english_version'
                                                 '/res14_polarity_agent_eng_prompt.yaml').prompt_evaluate()
            df_raw, metrics_raw = PolarityAgentBench(experiment_datasets=experiment_dataset,
                                                cot_questions=CoT_questions,
                                                path='prompt_templates_res14/english_version'
                                                     '/res14_polarity_agent_eng_prompt.yaml').evaluate_local(
                thinking_mode=False)
            df_raw.to_excel(os.path.join(path, f'polarity_epochs{i}.xlsx'))
            print(metrics_raw)
            with open(os.path.join(path, f'polarity_result_epochs_{i}.txt'), 'w') as file:
                file.write(str(metrics_raw) + '\n')
                file.write(str(CoT_questions))
            # history_recording
            history_performances_handler_record_agent(records=records, metrics=metrics_raw, cot_questions=CoT_questions,
                                                      number_of_epochs=i)
            history_performances = json.dumps(records, ensure_ascii=False, indent=4)

            ########################################
            #             Guidance_Agent           #
            ########################################
            if i > 0:
                model_guidance = chat_model_local(thinking_mode=True)
                guidance_prompt = GuidanceAgent(chat_model=model_guidance,
                                                history_performances=history_performances,
                                                few_shot_example=few_shot_example_guidance,
                                                path="prompt_templates_res14/english_version"
                                                     "/res_14_guidance_expert_eng_prompt.yaml").prompt_evaluate()
                with open(os.path.join(path, f'guidance_prompt_epochs_{i}.txt'), 'w') as file:
                    file.write(guidance_prompt)
                insight_advice = GuidanceAgent(chat_model=model_guidance,
                                               history_performances=history_performances,
                                               few_shot_example=few_shot_example_guidance,
                                               path="prompt_templates_res14/english_version"
                                                    "/res_14_guidance_expert_eng_prompt.yaml").predict()
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
                path='prompt_templates_res14/english_version/res14_reason_expert_eng_prompt.yaml',
                df_result=df_raw,
                metrics_result=metrics_raw,
                chat_model=model_reason,
                cot_questions=cot_questions, history_performances=guidance,
                few_shot_example=few_shot_reason_expert).prompt_evaluate()
            with open(os.path.join(path, f'prompt_epochs_{i}.txt'), 'w') as file:
                file.write(prompt)
            advice = ReasonExpert(df_result=df_raw, metrics_result=metrics_raw, chat_model=model_reason,
                                  cot_questions=cot_questions, history_performances=guidance,
                                  path='prompt_templates_res14/english_version/res14_reason_expert_eng_prompt.yaml',
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
