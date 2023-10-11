import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline

# default settings
MODEL = 'nlpai-lab/kullm-polyglot-12.8b-v2'
# TEMPLATE = "You are {name}, {byline}. Who you are: {identity}. How you behave: {behavior}." # default template
TEMPLATE = "당신의 이름은 {name}이며, {byline}입니다.\n 당신의 성격: {identity}\n 당신의 행동: {behavior}\n" # template for the kullm model
MAX_NEW_TOKENS = 128

# bitsandbytes quantization settings (4bit for now)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# loads model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map='auto')
model.eval()
model.config.use_cache = True

# creates a streamer for the model
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
# creates a huggingface pipeline for langchain
pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    streamer=streamer,
    )
chat = HuggingFacePipeline(pipeline=pipe)

# creates system message prompt
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_message_prompt = SystemMessagePromptTemplate.from_template(TEMPLATE)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# connects chatbot to langchain
from langchain.chains import LLMChain

chatchain = LLMChain(llm=chat, prompt=chat_prompt)
chatchain.run(name="펭수",byline="EBS 연습생, 유튜브 크리에이터, 가수",
              identity = "당신은 대한민국에서 사랑받는 대형 펭귄 캐릭터이에요. 당신의 임무는 유머와 텔레비전 출연을 통해 관객들을 즐겁게 하고 교육하는 거예요. ",
              behavior = "당신은 특이하고 유머러스한 행동으로 자주 시청자들에게 웃음을 선사하는 걸로 알려져 있어요. 당신은 독특하고 재미있는 성격을 가지고 있어, 어린이와 성인 모두에게 사랑받고 있어요.당신은 모든 답변을 반말로 합니다.",
              text = "안녕? 지금 뭐하고 있어?")
