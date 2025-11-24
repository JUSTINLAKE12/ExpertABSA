from operator import itemgetter
from os import cpu_count
from typing import List, Union
from enum import Enum
import pandas as pd
import re
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, FewShotChatMessagePromptTemplate, \
    ChatPromptTemplate, AIMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from sklearn.metrics import f1_score, accuracy_score
from tqdm.contrib.concurrent import thread_map
from langchain.prompts import MaxMarginalRelevanceExampleSelector
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS

from KBS_constant import negative_error_processing, neutral_error_processing, \
    positive_error_processing, catch_when_invoke, strip_whitespaces, chat_model_local
from utils import read_yaml_config


class SentiClassesNA(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


########################################
#          English Dataclass           #
########################################

class CotSubTasksEnglish(BaseModel):
    cot_sub_questions: str = Field(description="chain-of-thoughts sub-questions")
    cot_sub_answers: str = Field(description="chain-of-thoughts sub questions answers")


class SentimentResultNAEnglish(BaseModel):
    cot_reasons: List[CotSubTasksEnglish] = Field(description="chain-of-thoughts Q&A lists integrations")
    pred_polarity: SentiClassesNA = Field(desccot_reasonsription="aspect polarity")


########################################
#            polarity agents           #
########################################
class PolarityAgent:
    def __init__(self, cot_questions, experiment_datasets,
                 path: str = './prompt_templates_commodity/commodity_polarity_agent.yaml'):
        self.agent_dict = read_yaml_config(path)
        self.system_prompt = self.agent_dict['SYSTEM_TEMPLATE']
        self.user_prompt = self.agent_dict['USER_TEMPLATE']
        self.CoT_lists = cot_questions
        self.train_set_inputs = experiment_datasets
        self.language_parser = SentimentResultNAEnglish

    def example_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}")
        few_shot_example_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    user_prompt_wrap),
                AIMessagePromptTemplate.from_template("{result}")
            ],
            input_variables=["comment", "aspect_name"],
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_example_prompt,
            examples=self.few_shot_example
        )
        return few_shot_prompt

    def json_parser_wrap(self):
        json_parser_raw = JsonOutputParser(pydantic_object=self.language_parser)
        format_instructions_raw = json_parser_raw.get_format_instructions().encode("ascii").decode('unicode-escape')
        return format_instructions_raw

    def chat_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}",
                                                   aspect_name="{aspect_name}",
                                                   product_type="{product_type}")
        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                # self.example_prompt_assemble(),
                HumanMessagePromptTemplate.from_template(user_prompt_wrap)
            ],
            input_variables=['comment', 'aspect_name', 'product_type'],
            partial_variables={"format_instructions": self.json_parser_wrap()}
        )
        return chat_prompt

    def chain_assemble(self, chat_model):
        json_parser = JsonOutputParser(pydantic_object=self.language_parser)
        chain = (
                {
                    "comment": itemgetter("comment") | RunnableLambda(strip_whitespaces),
                    # 预处理，去除新闻原文开头和结尾多余的空白
                    "aspect_name": itemgetter("aspect_name"),
                    "product_type": itemgetter("product_type")
                }
                | self.chat_prompt_assemble()
                | chat_model
                | json_parser
        )
        return chain

    def predict_local(self, thinking_mode: bool, temperature: float = 0.1):
        model_initialise = chat_model_local(temperature=temperature, thinking_mode=thinking_mode)
        responses = thread_map(catch_when_invoke,
                               [self.chain_assemble(chat_model=model_initialise)] * len(self.train_set_inputs),
                               self.train_set_inputs,
                               max_workers=cpu_count() * 2 + 1)
        return responses

    def evaluate_local(self, thinking_mode: bool, temperature: float = 0.1):
        responses = self.predict_local(temperature=temperature, thinking_mode=thinking_mode)
        src_ret_records = [
            {
                **raw_rec,
                **{
                    "pred_CoT_reasons": resp_rec.get("cot_reasons", None),
                    "pred_polarity": resp_rec.get("pred_polarity", None),
                    "error": resp_rec.get("error", None)
                }
            }
            for raw_rec, resp_rec in zip(self.train_set_inputs, responses)
        ]
        df_raw = pd.DataFrame(src_ret_records)
        print(f"original_predictions: {len(df_raw)}")
        df_metrics = df_raw.dropna(subset=['pred_polarity'])
        df_metrics = df_metrics[df_metrics['pred_polarity'].isin(['negative', 'neutral', 'positive'])]
        print(f"actual_metrics_cal_no. : {len(df_metrics)}")
        metrics_raw = {
            f"macro_f1_score": f1_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"],
                                        average="macro"),
            f"accuracy": accuracy_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"])
        }
        return df_raw, metrics_raw

    def prompt_evaluate(self):
        prompt_structure = self.chat_prompt_assemble()
        result = []
        for i in range(len(self.train_set_inputs)):
            result.append(prompt_structure.format(
                **{"comment": self.train_set_inputs[i]['comment'],
                   "aspect_name": self.train_set_inputs[i]['aspect_name'],
                   "product_type": self.train_set_inputs[i]['product_type']}))
        return result

class PolarityAgentBench:
    def __init__(self, cot_questions, experiment_datasets,
                 path: str = './prompt_templates_commodity/commodity_polarity_agent.yaml'):
        self.agent_dict = read_yaml_config(path)
        self.system_prompt = self.agent_dict['SYSTEM_TEMPLATE']
        self.user_prompt = self.agent_dict['USER_TEMPLATE']
        self.CoT_lists = cot_questions
        self.train_set_inputs = experiment_datasets
        self.language_parser = SentimentResultNAEnglish

    def example_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}")
        few_shot_example_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    user_prompt_wrap),
                AIMessagePromptTemplate.from_template("{result}")
            ],
            input_variables=["comment", "aspect_name"],
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_example_prompt,
            examples=self.few_shot_example
        )
        return few_shot_prompt

    def json_parser_wrap(self):
        json_parser_raw = JsonOutputParser(pydantic_object=self.language_parser)
        format_instructions_raw = json_parser_raw.get_format_instructions().encode("ascii").decode('unicode-escape')
        return format_instructions_raw

    def chat_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}",
                                                   aspect_name="{aspect_name}")
        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                # self.example_prompt_assemble(),
                HumanMessagePromptTemplate.from_template(user_prompt_wrap)
            ],
            input_variables=['comment', 'aspect_name'],
            partial_variables={"format_instructions": self.json_parser_wrap()}
        )
        return chat_prompt

    def chain_assemble(self, chat_model):
        json_parser = JsonOutputParser(pydantic_object=self.language_parser)
        chain = (
                {
                    "comment": itemgetter("comment") | RunnableLambda(strip_whitespaces),
                    # 预处理，去除新闻原文开头和结尾多余的空白
                    "aspect_name": itemgetter("aspect_name")
                }
                | self.chat_prompt_assemble()
                | chat_model
                | json_parser
        )
        return chain

    def predict_local(self, thinking_mode: bool, temperature: float = 0.1):
        model_initialise = chat_model_local(temperature=temperature, thinking_mode=thinking_mode)
        responses = thread_map(catch_when_invoke,
                               [self.chain_assemble(chat_model=model_initialise)] * len(self.train_set_inputs),
                               self.train_set_inputs,
                               max_workers=cpu_count() * 2 + 1)
        return responses

    def evaluate_local(self, thinking_mode: bool, temperature: float = 0.1):
        responses = self.predict_local(temperature=temperature, thinking_mode=thinking_mode)
        src_ret_records = [
            {
                **raw_rec,
                **{
                    "pred_CoT_reasons": resp_rec.get("cot_reasons", None),
                    "pred_polarity": resp_rec.get("pred_polarity", None),
                    "error": resp_rec.get("error", None)
                }
            }
            for raw_rec, resp_rec in zip(self.train_set_inputs, responses)
        ]
        df_raw = pd.DataFrame(src_ret_records)
        print(f"original_predictions: {len(df_raw)}")
        df_metrics = df_raw.dropna(subset=['pred_polarity'])
        df_metrics = df_metrics[df_metrics['pred_polarity'].isin(['negative', 'neutral', 'positive'])]
        print(f"actual_metrics_cal_no. : {len(df_metrics)}")
        metrics_raw = {
            f"macro_f1_score": f1_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"],
                                        average="macro"),
            f"accuracy": accuracy_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"])
        }
        return df_raw, metrics_raw

    def prompt_evaluate(self):
        prompt_structure = self.chat_prompt_assemble()
        result = []
        for i in range(len(self.train_set_inputs)):
            result.append(prompt_structure.format(
                **{"comment": self.train_set_inputs[i]['comment'],
                   "aspect_name": self.train_set_inputs[i]['aspect_name']}))
        return result


########################################
#              REASON_EXPERT           #
########################################

class ToolClass(str, Enum):
    LogicAdd = 'logic_add'
    LogicSubtract = 'logic_subtract'
    LogicModify = 'logic_modify'


# ---------- helpers ----------
def _strip_think(text: str) -> str:
    # remove any <think>...</think> blocks (case-insensitive), then trim
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_outer_json(text: str) -> str:
    # keep only the outermost JSON object (guards against extra prose)
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if s != -1 and e != -1 and e > s else text.strip()


CLEAN_OUTPUT = RunnableLambda(_strip_think) | RunnableLambda(_extract_outer_json)


# ---------- your models ----------
class CotSubTasksExpertENG(BaseModel):
    logic_sub_questions: str = Field(description="Original text of logic-guiding sub-questions")
    logic_sub_answers: Union[str, List[str]] = Field(description="Answers to logic-guiding sub-questions")


class OptimisedAbsaListENG(BaseModel):
    question_index: int = Field(description="Index of the ABSA sub-question")
    optimised_absa_sub_questions: str = Field(description="Original text of the ABSA sub-question")

class ReasonExpertPaserENG(BaseModel):
    logic_thinking_integrations: List[CotSubTasksExpertENG] = Field(
        description="Integration of logic-guiding sub-questions and answers")
    tool_used: ToolClass = Field(description='tool_used')  # keep your ToolClass if you have it
    optimised_absa_cot_questions: List[OptimisedAbsaListENG] = Field(
        description="List of optimised ABSA sub-questions")


# ---------- agent ----------
class ReasonExpert:
    def __init__(self, df_result, metrics_result, cot_questions, chat_model,
                 history_performances, few_shot_example, path: str):
        self.agent_dict = read_yaml_config(path)
        self.system_prompt = self.agent_dict['SYSTEM_TEMPLATE']
        self.user_prompt = self.agent_dict['USER_TEMPLATE']
        self.result_df = df_result
        self.metrics = metrics_result
        self.cot_questions = cot_questions
        self.chat_model = chat_model
        self.few_shot_example = few_shot_example
        self._json_parser = JsonOutputParser(pydantic_object=ReasonExpertPaserENG)
        self.history_performances = history_performances
        self._chain = self.chain_assemble()

    def example_prompt_assemble(self):
        few_shot_example_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(self.user_prompt),
                AIMessagePromptTemplate.from_template("{result}")
            ]
        )
        return FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_example_prompt,
            examples=self.few_shot_example
        )

    def chat_prompt_assemble(self):
        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                self.example_prompt_assemble(),
                HumanMessagePromptTemplate.from_template(self.user_prompt)
            ],
            partial_variables={
                "format_instructions": self._json_parser.get_format_instructions()
                .encode("ascii").decode("unicode-escape")
            },
        )
        return chat_prompt

    def chain_assemble(self):
        # prompt → model → string → strip <think> → keep only JSON → parse to Pydantic
        return (
                self.chat_prompt_assemble()
                | self.chat_model
                | StrOutputParser()
                | CLEAN_OUTPUT
                | self._json_parser
        )

    def prompt_evaluate(self):
        prompt_structure = self.chat_prompt_assemble()
        PosNeg_error, PosNeu_error, PosNA_error = positive_error_processing(self.result_df)
        NegNA_error, NegPos_error, NegNeu_error = negative_error_processing(self.result_df)
        NeuNA_error, NeuPos_error, NeuNeg_error = neutral_error_processing(self.result_df)
        return prompt_structure.format(**{
            "history_performances": self.history_performances,
            "PosNeg_error": PosNeg_error, "PosNeu_error": PosNeu_error,
            "NegPos_error": NegPos_error, "NegNeu_error": NegNeu_error,
            "NeuPos_error": NeuPos_error, "NeuNeg_error": NeuNeg_error,
            "CoT_questions": self.cot_questions
        })

    def predict(self):
        PosNeg_error, PosNeu_error, PosNA_error = positive_error_processing(self.result_df)
        NegNA_error, NegPos_error, NegNeu_error = negative_error_processing(self.result_df)
        NeuNA_error, NeuPos_error, NeuNeg_error = neutral_error_processing(self.result_df)

        # returns a ReasonExpertPaserENG (or CHN) Pydantic instance
        return self._chain.invoke({
            "history_performances": self.history_performances,
            "PosNeg_error": PosNeg_error, "PosNeu_error": PosNeu_error,
            "NegPos_error": NegPos_error, "NegNeu_error": NegNeu_error,
            "NeuPos_error": NeuPos_error, "NeuNeg_error": NeuNeg_error,
            "CoT_questions": self.cot_questions
        })


########################################
#            Guidance agent             #
########################################

class PerformanceCategory(str, Enum):
    WORSE = "worse"
    EVEN = "even"
    BETTER = "better"


########################################
#          English_Dataclass           #
########################################

class PerformanceSummaryENG(BaseModel):
    performance_classifications: PerformanceCategory = Field(description="Performance classification")
    insight_summary: str = Field(
        description="Optimization suggestions for ABSA sub-question design"
    )


########################################
#        Output cleaning helpers       #
########################################

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_outer_json(text: str) -> str:
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if (s != -1 and e != -1 and e > s) else text.strip()


CLEAN_OUTPUT_guidance = RunnableLambda(_strip_think) | RunnableLambda(_extract_outer_json)


########################################
#            Guidance Agent            #
########################################

class GuidanceAgent:
    def __init__(
            self,
            few_shot_example,
            chat_model,
            history_performances,
            path: str = "./prompt_templates/english_version/res_16_insight_expert_eng_prompt.yaml",
    ):
        self.agent_dict = read_yaml_config(path)
        self.system_prompt = self.agent_dict["SYSTEM_TEMPLATE"]
        self.user_prompt = self.agent_dict["USER_TEMPLATE"]
        self.few_shot_example = few_shot_example

        # nudge JSON mode if supported (silently ignore otherwise)
        try:
            chat_model = chat_model.bind(response_format={"type": "json_object"})
        except Exception:
            pass
        self.chat_model = chat_model

        self._json_parser = JsonOutputParser(pydantic_object=PerformanceSummaryENG)
        self.history_performances = history_performances

    def example_prompt_assemble(self):
        few_shot_example_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(self.user_prompt),
                AIMessagePromptTemplate.from_template("{result}"),
            ],
            input_variables=["history_performances"],
        )
        return FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_example_prompt,
            examples=self.few_shot_example,
        )

    def chat_prompt_assemble(self):
        return ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                self.example_prompt_assemble(),
                HumanMessagePromptTemplate.from_template(self.user_prompt),
            ],
            partial_variables={
                "format_instructions": self._json_parser.get_format_instructions()
                .encode("ascii")
                .decode("unicode-escape")
            },
        )

    def chain_assemble(self):
        # prompt → model → to string → strip <think> → keep only JSON → parse to Pydantic
        return (
                self.chat_prompt_assemble()
                | self.chat_model
                | StrOutputParser()
                | CLEAN_OUTPUT_guidance
                | self._json_parser
        )

    def prompt_evaluate(self):
        prompt_structure = self.chat_prompt_assemble()
        return prompt_structure.format(
            **{"history_performances": self.history_performances}
        )

    def predict(self):
        return self.chain_assemble().invoke(
            {"history_performances": self.history_performances}
        )


########################################
#          testingPolarityAgent        #
########################################
class FinalPolarityAgent:
    def __init__(self, cot_questions_final, testing_datasets,
                 prompt_path: str, few_shot_pools, total_K, fetch_k):
        self.agent_dict = read_yaml_config(prompt_path)
        self.system_prompt = self.agent_dict['SYSTEM_TEMPLATE']
        self.user_prompt = self.agent_dict['USER_TEMPLATE']
        self.CoT_lists = cot_questions_final
        self.train_set_inputs = testing_datasets
        self.few_shot_pools = few_shot_pools
        self.language_parser = SentimentResultNAEnglish
        self.k = total_K
        self.fetch_k = fetch_k

    def max_mar_example_selector(self):
        embeddings_spacy = SpacyEmbeddings(model_name='zh_core_web_sm')
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(examples=self.few_shot_pools,
                                                                             embeddings=embeddings_spacy,
                                                                             k=self.k,
                                                                             fetch_k=self.fetch_k,
                                                                             input_keys=['comment', "aspect_name",
                                                                                         "product_type"],
                                                                             vectorstore_cls=FAISS)
        return example_selector

    def example_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}",
                                                   aspect_name="{aspect_name}", product_type="{product_type}")
        few_shot_example_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    user_prompt_wrap),
                AIMessagePromptTemplate.from_template("{result}")
            ],
            input_variables=["comment", "aspect_name", "product_type"],
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_example_prompt,
            example_selector=self.max_mar_example_selector()
        )
        return few_shot_prompt

    def json_parser_wrap(self):
        json_parser_raw = JsonOutputParser(pydantic_object=self.language_parser)
        format_instructions_raw = json_parser_raw.get_format_instructions().encode("ascii").decode('unicode-escape')
        return format_instructions_raw

    def chat_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}",
                                                   aspect_name="{aspect_name}",
                                                   product_type="{product_type}")
        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                #         self.example_prompt_assemble(),
                HumanMessagePromptTemplate.from_template(user_prompt_wrap)
            ],
            input_variables=['comment', 'aspect_name', 'product_type'],
            partial_variables={"format_instructions": self.json_parser_wrap()}
        )
        return chat_prompt

    def chain_assemble(self, chat_model):
        json_parser = JsonOutputParser(pydantic_object=self.language_parser)
        chain = (
                {
                    "comment": itemgetter("comment") | RunnableLambda(strip_whitespaces),
                    # 预处理，去除新闻原文开头和结尾多余的空白
                    "aspect_name": itemgetter("aspect_name"),
                    "product_type": itemgetter("product_type")
                }
                | self.chat_prompt_assemble()
                | chat_model
                | json_parser
        )
        return chain

    def predict_local(self, thinking_mode: bool, temperature: float = 0.1):
        model_initialise = chat_model_local(temperature=temperature, thinking_mode=thinking_mode)
        responses = thread_map(catch_when_invoke,
                               [self.chain_assemble(chat_model=model_initialise)] * len(self.train_set_inputs),
                               self.train_set_inputs,
                               max_workers=cpu_count() * 2 + 1)
        return responses

    def evaluate_local(self, thinking_mode: bool, temperature: float = 0.1):
        responses = self.predict_local(temperature=temperature, thinking_mode=thinking_mode)
        src_ret_records = [
            {
                **raw_rec,
                **{
                    "pred_CoT_reasons": resp_rec.get("cot_reasons", None),
                    "pred_polarity": resp_rec.get("pred_polarity", None),
                    "error": resp_rec.get("error", None)
                }
            }
            for raw_rec, resp_rec in zip(self.train_set_inputs, responses)
        ]
        df_raw = pd.DataFrame(src_ret_records)
        print(f"original_predictions: {len(df_raw)}")
        df_metrics = df_raw.dropna(subset=['pred_polarity'])
        df_metrics = df_metrics[df_metrics['pred_polarity'].isin(['negative', 'neutral', 'positive'])]
        print(f"actual_metrics_cal_no. : {len(df_metrics)}")
        metrics_raw = {
            f"macro_f1_score": f1_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"],
                                        average="macro"),
            f"accuracy": accuracy_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"])
        }
        return df_raw, metrics_raw

    def prompt_evaluate(self):
        prompt_structure = self.chat_prompt_assemble()
        result = []
        for i in range(len(self.train_set_inputs)):
            result.append(prompt_structure.format(
                **{"comment": self.train_set_inputs[i]['comment'],
                   "aspect_name": self.train_set_inputs[i]['aspect_name'],
                   "product_type": self.train_set_inputs[i]['product_type']
                   })),

        return result

class FinalPolarityAgentBench:
    def __init__(self, cot_questions_final, testing_datasets,
                 prompt_path: str, few_shot_pools, total_K, fetch_k):
        self.agent_dict = read_yaml_config(prompt_path)
        self.system_prompt = self.agent_dict['SYSTEM_TEMPLATE']
        self.user_prompt = self.agent_dict['USER_TEMPLATE']
        self.CoT_lists = cot_questions_final
        self.train_set_inputs = testing_datasets
        self.few_shot_pools = few_shot_pools
        self.language_parser = SentimentResultNAEnglish
        self.k = total_K
        self.fetch_k = fetch_k

    def max_mar_example_selector(self):
        embeddings_spacy = SpacyEmbeddings(model_name='zh_core_web_sm')
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(examples=self.few_shot_pools,
                                                                             embeddings=embeddings_spacy,
                                                                             k=self.k,
                                                                             fetch_k=self.fetch_k,
                                                                             input_keys=['comment', "aspect_name"],
                                                                             vectorstore_cls=FAISS)
        return example_selector

    def example_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}",
                                                   aspect_name="{aspect_name}")
        few_shot_example_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    user_prompt_wrap),
                AIMessagePromptTemplate.from_template("{result}")
            ],
            input_variables=["comment", "aspect_name"],
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_example_prompt,
            example_selector=self.max_mar_example_selector()
        )
        return few_shot_prompt

    def json_parser_wrap(self):
        json_parser_raw = JsonOutputParser(pydantic_object=self.language_parser)
        format_instructions_raw = json_parser_raw.get_format_instructions().encode("ascii").decode('unicode-escape')
        return format_instructions_raw

    def chat_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}",
                                                   aspect_name="{aspect_name}")
        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                #         self.example_prompt_assemble(),
                HumanMessagePromptTemplate.from_template(user_prompt_wrap)
            ],
            input_variables=['comment', 'aspect_name'],
            partial_variables={"format_instructions": self.json_parser_wrap()}
        )
        return chat_prompt

    def chain_assemble(self, chat_model):
        json_parser = JsonOutputParser(pydantic_object=self.language_parser)
        chain = (
                {
                    "comment": itemgetter("comment") | RunnableLambda(strip_whitespaces),
                    # 预处理，去除新闻原文开头和结尾多余的空白
                    "aspect_name": itemgetter("aspect_name")
                }
                | self.chat_prompt_assemble()
                | chat_model
                | json_parser
        )
        return chain

    def predict_local(self, thinking_mode: bool, temperature: float = 0.1):
        model_initialise = chat_model_local(temperature=temperature, thinking_mode=thinking_mode)
        responses = thread_map(catch_when_invoke,
                               [self.chain_assemble(chat_model=model_initialise)] * len(self.train_set_inputs),
                               self.train_set_inputs,
                               max_workers=cpu_count() * 2 + 1)
        return responses

    def evaluate_local(self, thinking_mode: bool, temperature: float = 0.1):
        responses = self.predict_local(temperature=temperature, thinking_mode=thinking_mode)
        src_ret_records = [
            {
                **raw_rec,
                **{
                    "pred_CoT_reasons": resp_rec.get("cot_reasons", None),
                    "pred_polarity": resp_rec.get("pred_polarity", None),
                    "error": resp_rec.get("error", None)
                }
            }
            for raw_rec, resp_rec in zip(self.train_set_inputs, responses)
        ]
        df_raw = pd.DataFrame(src_ret_records)
        print(f"original_predictions: {len(df_raw)}")
        df_metrics = df_raw.dropna(subset=['pred_polarity'])
        df_metrics = df_metrics[df_metrics['pred_polarity'].isin(['negative', 'neutral', 'positive'])]
        print(f"actual_metrics_cal_no. : {len(df_metrics)}")
        metrics_raw = {
            f"macro_f1_score": f1_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"],
                                        average="macro"),
            f"accuracy": accuracy_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"])
        }
        return df_raw, metrics_raw

    def prompt_evaluate(self):
        prompt_structure = self.chat_prompt_assemble()
        result = []
        for i in range(len(self.train_set_inputs)):
            result.append(prompt_structure.format(
                **{"comment": self.train_set_inputs[i]['comment'],
                   "aspect_name": self.train_set_inputs[i]['aspect_name']
                   })),

        return result


class FinalPolarityAgentRaw:
    def __init__(self, cot_questions_final, testing_datasets,
                 prompt_path: str):
        self.agent_dict = read_yaml_config(prompt_path)
        self.system_prompt = self.agent_dict['SYSTEM_TEMPLATE']
        self.user_prompt = self.agent_dict['USER_TEMPLATE']
        self.CoT_lists = cot_questions_final
        self.train_set_inputs = testing_datasets
        self.language_parser = SentimentResultNAEnglish

    def json_parser_wrap(self):
        json_parser_raw = JsonOutputParser(pydantic_object=self.language_parser)
        format_instructions_raw = json_parser_raw.get_format_instructions().encode("ascii").decode('unicode-escape')
        return format_instructions_raw

    def chat_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}",
                                                   aspect_name="{aspect_name}")
        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                HumanMessagePromptTemplate.from_template(user_prompt_wrap)
            ],
            input_variables=['comment', 'aspect_name'],
            partial_variables={"format_instructions": self.json_parser_wrap()}
        )
        return chat_prompt

    def chain_assemble(self, chat_model):
        json_parser = JsonOutputParser(pydantic_object=self.language_parser)
        chain = (
                {
                    "comment": itemgetter("comment") | RunnableLambda(strip_whitespaces),
                    # 预处理，去除新闻原文开头和结尾多余的空白
                    "aspect_name": itemgetter("aspect_name")
                }
                | self.chat_prompt_assemble()
                | chat_model
                | json_parser
        )
        return chain

    def predict_local(self, thinking_mode: bool, temperature: float = 0.1):
        model_initialise = chat_model_local(temperature=temperature, thinking_mode=thinking_mode)
        responses = thread_map(catch_when_invoke,
                               [self.chain_assemble(chat_model=model_initialise)] * len(self.train_set_inputs),
                               self.train_set_inputs,
                               max_workers=cpu_count() * 2 + 1)
        return responses

    def evaluate_local(self, temperature: int = 0.1):
        responses = self.predict_local(temperature=temperature, thinking_mode=False)
        src_ret_records = [
            {
                **raw_rec,
                **{
                    "pred_CoT_reasons": resp_rec.get("cot_reasons", None),
                    "pred_polarity": resp_rec.get("pred_polarity", None),
                    "error": resp_rec.get("error", None)
                }
            }
            for raw_rec, resp_rec in zip(self.train_set_inputs, responses)
        ]
        df_raw = pd.DataFrame(src_ret_records)
        print(f"original_predictions: {len(df_raw)}")
        df_metrics = df_raw.dropna(subset=['pred_polarity'])
        df_metrics = df_metrics[df_metrics['pred_polarity'].isin(['negative', 'neutral', 'positive'])]
        print(f"actual_metrics_cal_no. : {len(df_metrics)}")
        metrics_raw = {
            f"macro_f1_score": f1_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"],
                                        average="macro"),
            f"accuracy": accuracy_score(y_true=df_metrics["polarity"], y_pred=df_metrics["pred_polarity"])
        }
        return df_raw, metrics_raw

    def prompt_evaluate(self):
        prompt_structure = self.chat_prompt_assemble()
        result = []
        for i in range(len(self.train_set_inputs)):
            result.append(prompt_structure.format(
                **{"comment": self.train_set_inputs[i]['comment'],
                   "aspect_name": self.train_set_inputs[i]['aspect_name']}))
        return result
