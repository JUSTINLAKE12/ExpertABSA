import os
import json
import pandas as pd
from Agent_pools import FinalPolarityAgent
from sklearn.metrics import f1_score, accuracy_score

########################################
#           Data Processing            #
########################################

df = pd.read_json('./commodity_dataset/commodity25_test.json')

def pick_first_nonempty(row):
    for col in ['positive_sentences', 'neutral_sentences', 'negative_sentences']:
        val = row[col]
        if val not in [None, [], '', 'nan']:
            return val
    return None

df['comment'] = df.apply(pick_first_nonempty, axis=1)

dataset = pd.DataFrame({
    'comment': df['comment'],
    'product_type': df['product_type'],
    'aspect_name': df['aspect_name'],
    'polarity': df['sentiment']
}).to_dict(orient='records')

few_shots_pools = pd.read_json('testing_comm25/few_shots_pools.json').to_dict(orient='records')

########################################
#         CoT Prompt Variants          #
########################################

cot_variants = {
    "cot_full": """
- Q1: Are there explicit sentiment indicators (e.g., 'increase', 'decline') in the text related to {aspect_name}, and how does their contextual usage affect the aspect in this commodity scenario (positive/negative/neutral)?
- Q2: Is the {aspect_name} directly mentioned in the commodity news or implied through metrics that directly affect the aspect?
- Q3: What is the intensity of the sentiment indicator (strong/moderate/weak)?
- Q4: Based on Q1-Q3, what is the sentiment polarity of {aspect_name} in this commodity news?
""",

    "cot_ablation_no_absa": """
# - Q1: Are there indicators (e.g., 'increase', 'decline') in the text related to {aspect_name}?
# - Q2: What is the intensity of the sentiment indicator (strong/moderate/weak)?
# - Q3: Based on Q1-Q2, what is the sentiment polarity of {aspect_name} in this commodity news?
""",

    "cot_ablation_no_domain": """
# - Q1: Are there explicit sentiment indicators in the text related to {aspect_name}?
# - Q2: Is the {aspect_name} directly mentioned in the text or implied through metrics that directly affect the aspect?
# - Q3: Based on Q1-Q2, what is the sentiment polarity of {aspect_name} in this text?
""",

    "cot_raw": """What is the sentiment polarity of {aspect_name} in this commodity news?"""
}


########################################
#       Evaluation Wrapper Function     #
########################################

def evaluate_one_cot(cot_name, cot_prompt, epoch_id):
    """
    Run one CoT type for one epoch.
    If cot_name == "cot_full":
        fetch_k = 8, total_K = 20
    Otherwise:
        fetch_k = 0, total_K = 0
    """

    # Determine setting based on whether ABSA fully enabled
    if cot_name == "cot_full":
        fetch_k = 8
        total_K = 12
    else:
        fetch_k = 0
        total_K = 0

    save_dir = f'./comm25_result&ablation_study/{cot_name}/epoch_{epoch_id}'
    os.makedirs(save_dir, exist_ok=True)

    model = FinalPolarityAgent(
        cot_questions_final=cot_prompt,
        testing_datasets=dataset,
        fetch_k=fetch_k,
        total_K=total_K,
        prompt_path='prompt_templates_comm25/commodity_polarity_agent.yaml',
        few_shot_pools=few_shots_pools
    )

    prompt_used = model.prompt_evaluate()
    result_df, metrics_test = model.evaluate_local(thinking_mode=False)

    # Save result df
    result_df['prompt'] = prompt_used
    result_df.to_excel(os.path.join(save_dir, f"result_{cot_name}.xlsx"), index=False)

    # Save metrics + prompting template
    with open(os.path.join(save_dir, f"metrics_{cot_name}.txt"), "w") as f:
        f.write(str(metrics_test) + "\n")
        f.write(cot_prompt)

    print(f"[DONE] {cot_name} | epoch {epoch_id}")
    print(f"fetch_k={fetch_k}, total_K={total_K}")
    print(f"Metrics: {metrics_test}\n")


########################################
#              MAIN LOOP               #
########################################

if __name__ == "__main__":

    for epoch in range(3):  # epochs: 0, 1, 2
        print(f"======================")
        print(f" Running epoch {epoch} ")
        print(f"======================")

        for cot_name, cot_prompt in cot_variants.items():
            evaluate_one_cot(cot_name, cot_prompt, epoch)
