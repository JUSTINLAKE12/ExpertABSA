# ExpertABSA for Aspect-based Sentiment Analysis

This repository contains the implementation of ExpertABSA for Aspect-Based Sentiment Analysis (ABSA), using chain-of-thought reasoning and multi-agent framework to optimize sentiment polarity prediction.

## Environment Setup

1. Python 3.10
2. Install dependencies using: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file:
   - `MODEL_NAME`: Your LLM model name
   - `BASE_URL`: API endpoint for your LLM
   - `API_KEY`: Your API key
   - `EPOCHS`: Number of training epochs (default: 8)
   - `EXPERIMENT_PATH`: Path for experiment outputs (default: 'result')

### To run training stage I of ExpertABSA:

Before running any experiments, ensure your environment variables are properly set in the `.env` file.

#### For RES14 (Restaurant) dataset:
- Run `Experiment_Res_ENG.py` to implement ExpertABSA ExpertABSA stage I training  process.

#### For LAP14 (Laptop) dataset:
- Run `Experiment_lap_ENG.py` to implement ExpertABSA ExpertABSA stage I training  process.

#### For commodity dataset:
- Run `commodity_experiment.py` to implement ExpertABSA ExpertABSA stage I training  process.

The training process involves three main agents working iteratively:
1. **Polarity Agent**: Predicts sentiment polarity using chain-of-thought questions
2. **Guidance Agent**: Analyzes performance history and provides optimization insights
3. **Reason Expert**: Analyzes error patterns and optimizes the chain-of-thought questions

### To run training stage II (Error Correction) of ExpertABSA:

Before running training stage II, you need to:
1. Obtain the optimized CoT questions ("cot_final") from training stage I
2. Prepare the corresponding error dataset ("result_df") from training stage I
3. Design hand-crafted examples for the error correction process

Run `Experiment2_error_corrections.py` with the required inputs to implement the error correction mechanism.

### To directly run testing stage and ablation studies of ExpertABSA:
1. for lap_14, run `testing_lap_14_english.py`
2. for res_14, run `evaluation_res14_english.py`
3. for comm_25, run `commodity_test.py`

### Project Structure:

- `Agent_pools.py`: Contains the implementation of all agents (PolarityAgent, GuidanceAgent, ReasonExpert, etc.)
- `KBS_constant.py`: Contains utility functions, error processing functions, and data handling utilities
- `utils.py`: Utility functions for reading YAML configuration files
- `few_shots_example_agent.py`: Contains example implementations for few-shot learning
- `prompt_templates_*/`: Directory containing prompt templates for different datasets
- `RES14/`: Restaurant dataset files
- `LAP14/`: Laptop dataset files
- `commodity_dataset/`: Commodity dataset files





