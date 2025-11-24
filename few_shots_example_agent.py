from enum import Enum
from operator import itemgetter
from os import cpu_count
from typing import List

from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, FewShotChatMessagePromptTemplate, \
    ChatPromptTemplate, AIMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from tqdm.contrib.concurrent import thread_map

from KBS_constant import chat_model_local
from Agent_pools import catch_when_invoke, strip_whitespaces, chat_model_alibaba
from utils import read_yaml_config


class SentiClassesNA(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class CotSubTasks(BaseModel):
    cot_sub_questions: str = Field(description="chain-of-thoughts sub-questions")
    cot_sub_answers: str = Field(description="chain-of-thoughts sub questions answers")


class FewShotOutput(BaseModel):
    ground_truth_polarity: SentiClassesNA = Field(description="ground truth polarity")
    cot_reasons: List[CotSubTasks] = Field(description="chain-of-thoughts Q&A lists integrations")
    pred_polarity: SentiClassesNA = Field(desccot_reasonsription="cot reasons reasoning polarity")


class ErrorCorrectionAgent:
    def __init__(self, few_shot_example, cot_questions_final, cot_questions_result_df,
                 path: str):
        self.agent_dict = read_yaml_config(path)
        self.system_prompt = self.agent_dict['SYSTEM_TEMPLATE']
        self.user_prompt = self.agent_dict['USER_TEMPLATE']
        self.few_shot_example = few_shot_example
        self.CoT_lists = cot_questions_final
        self.train_set_inputs = cot_questions_result_df

    def example_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}",
                                                   aspect_name="{aspect_name}", polarity="{polarity}", product_type = "{product_type}")
        few_shot_example_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(user_prompt_wrap),
                AIMessagePromptTemplate.from_template("{result}")
            ],
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_example_prompt,
            examples=self.few_shot_example
        )
        return few_shot_prompt

    def json_parser_wrap(self):
        json_parser_raw = JsonOutputParser(pydantic_object=FewShotOutput)
        format_instructions_raw = json_parser_raw.get_format_instructions().encode("ascii").decode('unicode-escape')
        return format_instructions_raw

    def chat_prompt_assemble(self):
        user_prompt_wrap = self.user_prompt.format(CoT_questions=self.CoT_lists, comment="{comment}",
                                                   aspect_name="{aspect_name}", polarity="{polarity}",product_type = "{product_type}")
        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
             #   self.example_prompt_assemble(),
                HumanMessagePromptTemplate.from_template(user_prompt_wrap)
            ],
            input_variables=["comment", "aspect_name", "polarity","{product_type}"],
            partial_variables={"format_instructions": self.json_parser_wrap()}
        )
        return chat_prompt

    def chain_assemble(self, chat_model):
        json_parser = JsonOutputParser(pydantic_object=FewShotOutput)

        chain = (
                {
                    "comment": itemgetter("comment") | RunnableLambda(strip_whitespaces),
                    "aspect_name": itemgetter("aspect_name"),
                    "polarity": itemgetter("polarity"),
                    'product_type': itemgetter("product_type")
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

