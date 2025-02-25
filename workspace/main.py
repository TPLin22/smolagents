from smolagents import tool, LiteLLMModel, HfApiModel, CodeAgent
from smolagents import DuckDuckGoSearchTool, GoogleSearchTool, VisitWebpageTool, FinalAnswerTool
from smolagents import PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
from typing import Optional
from dotenv import load_dotenv
import os
from cozepy import Coze, TokenAuth, Message, ChatStatus, COZE_CN_BASE_URL


load_dotenv()

HF_Token = os.getenv("HUGGINGFACE_TOKEN")
ds_api_token = os.getenv("DEEPSEEK_API_TOKEN")

# 配置 LLM 模型

# model is "Qwen/Qwen2.5-Coder-32B-Instruct" by default, can be set by model_id
model = HfApiModel(token=HF_Token) 
# model = LiteLLMModel(model_id="deepseek/deepseek-chat", api_key=ds_api_token)

from pathlib import Path

def load_markdown_file(file_path):
    """
    读取本地 Markdown 文件内容。
    Args:
        file_path: Markdown 文件路径
    Returns:
        文件内容
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content

# 加载 a.md 文件
md_content = load_markdown_file("research_report.md")

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 将 Markdown 内容分割为块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个块的大小
    chunk_overlap=50,  # 块之间的重叠部分
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# 将 Markdown 内容转换为 Document 对象
docs_processed = text_splitter.split_documents([Document(page_content=md_content)])

from langchain_community.retrievers import BM25Retriever
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "使用语义搜索从本地文档中检索与查询最相关的内容。该文档的内容是关于国际能源价格对粮食供应链的动态影响研究的，可以为粮食供应链的市场分析、供应链优化提供建议。该工具每个Agent只需要调用一次."
    inputs = {
        "query": {
            "type": "string",
            "description": "要执行的查询。应与目标文档语义接近。使用肯定形式而非问题形式。",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=1)  # 检索前 1 个相关文档

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "查询必须为字符串"

        docs = self.retriever.invoke(query)
        return "\n检索到的文档：\n" + "".join(
            [
                f"\n\n===== 文档 {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# 初始化检索工具
retriever_tool = RetrieverTool(docs_processed)

@tool
def get_market_analysis_report_format() -> str:
    """
    在market_analyst最终输出报告前，获取其输出市场分析报告的格式模板。
    
    Returns:
        市场分析报告的格式模板
    """
    try:
        with open('report_format/market_analysis_report_format.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "未找到市场分析报告格式文件，请确保文件 'market_analysis_report_format.txt' 存在。"
    except Exception as e:
        return f"读取市场分析报告格式文件时发生错误：{str(e)}"
    
@tool
def get_tech_analysis_report_format() -> str:
    """
    在tech_analyst最终输出报告前，获取其输出技术分析报告的格式模板。
    Returns:
        技术分析报告的格式模板
    """
    try:
        with open('report_format/tech_analysis_report_format.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "未找到技术分析报告格式文件，请确保文件 'tech_analysis_report_format.txt' 存在。"
    except Exception as e:
        return f"读取技术分析报告格式文件时发生错误：{str(e)}"
    
@tool
def get_supply_chain_analysis_report_format() -> str:
    """
    在manager输出supply chain报告前，获取其输出供应链分析报告的格式模板。
    Returns:
        供应链分析报告的格式模板
    """
    try:
        with open('report_format/supply_chain_report_format.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "未找到供应链分析报告格式文件，请确保文件 'supply_chain_report_format.txt' 存在。"
    except Exception as e:
        return f"读取供应链分析报告格式文件时发生错误：{str(e)}"
    

# The default access is api.coze.com, but if you need to access api.coze.cn,
# please use base_url to configure the api endpoint to access
# coze_api_base = os.getenv("COZE_API_BASE") or COZE_COM_BASE_URL

# Init the Coze client through the access_token.
@tool
def market_data_search(question: str) -> str:
    """
    根据提问，搜索市场相关数据（如价格、供需趋势等）。不要把问题拆解成太多来提问。
    不要使用此工具超过三次.
    Args:
        question: 要搜索的市场问题
    Returns:
        市场相关数据信息
    """
    coze = Coze(auth=TokenAuth(token=os.getenv("COZE_API_TOKEN")), base_url=COZE_CN_BASE_URL)

    # 创建并调用 Coze 聊天机器人
    chat_poll = coze.chat.create_and_poll(
        bot_id='7474855849769402419',  # 替换为你的市场数据搜索机器人 ID
        user_id='user_id',  # 替换为你的用户 ID
        additional_messages=[Message.build_user_question_text(question)]
    )

    # 提取并返回聊天结果
    ans = ""
    for message in chat_poll.messages:
        ans += message.content
    return ans

@tool
def policy_search(question: str) -> str:
    """
    根据提问，搜索与市场相关的政策信息。
    不要使用此工具超过三次.
    Args:
        question: 要搜索的政策问题
    Returns:
        政策相关信息
    """
    coze = Coze(auth=TokenAuth(token=os.getenv("COZE_API_TOKEN")), base_url=COZE_CN_BASE_URL)

    # 创建并调用 Coze 聊天机器人
    chat_poll = coze.chat.create_and_poll(
        bot_id='7474855849769402419',  # 替换为你的政策搜索机器人 ID
        user_id='user_id',  # 替换为你的用户 ID
        additional_messages=[Message.build_user_question_text(question)]
    )

    # 提取并返回聊天结果
    ans = ""
    for message in chat_poll.messages:
        ans += message.content
    return ans


@tool
def tech_search(question: str) -> str:
    """
    根据提问，输出有关发展技术的新闻、文献信息。
    不要使用此工具超过两次.
    Args:
        question: 要搜寻的技术问题
    Returns:
        有关技术的新闻、文献信息
    """
    coze = Coze(auth=TokenAuth(token=os.getenv("COZE_API_TOKEN")), base_url=COZE_CN_BASE_URL)

# Create a bot instance in Coze, copy the last number from the web link as the bot's ID.

    chat_poll = coze.chat.create_and_poll(
        # id of bot
        bot_id='7473795915066294284',
        # id of user, Note: The user_id here is specified by the developer, for example, it can be the
        # business id in the developer system, and does not include the internal attributes of coze.
        user_id='user_id',
        # user input
        additional_messages=[Message.build_user_question_text(question)]
    )
    ans = ""
    for message in chat_poll.messages:
        ans += message.content
    return ans

@tool
def supply_chain_cases_search(question: str) -> str:
    """
    搜索供应链优化的有关经验以及国际大型粮食企业的成功案例
    不要使用此工具超过两次.
    Args:
        question: 要搜寻的技术问题
    Returns:
        供应链优化分析以及成功案例报告
    """
    coze = Coze(auth=TokenAuth(token=os.getenv("COZE_API_TOKEN")), base_url=COZE_CN_BASE_URL)

# Create a bot instance in Coze, copy the last number from the web link as the bot's ID.

    chat_poll = coze.chat.create_and_poll(
        # id of bot
        bot_id='7475034370354053156',
        # id of user, Note: The user_id here is specified by the developer, for example, it can be the
        # business id in the developer system, and does not include the internal attributes of coze.
        user_id='user_id',
        # user input
        additional_messages=[Message.build_user_question_text(question)]
    )
    ans = ""
    for message in chat_poll.messages:
        ans += message.content
    return ans

final_answer = FinalAnswerTool()

# 创建市场分析专家 Agent
market_analyst = CodeAgent(
    tools=[market_data_search, policy_search, get_market_analysis_report_format, retriever_tool, final_answer],
    model=model,
    name="market_analyst",
    description="市场分析专家，通过分析国际能源与粮食政策、国际能源市场、国际粮食市场给出我国玉米市场生产与贸易的相关分析结果。",
    additional_authorized_imports=["time", "numpy", "pandas"],
    max_steps=10,
)


tech_analyst = CodeAgent(
    tools=[tech_search, get_tech_analysis_report_format, final_answer],
    model=model,
    name="tech_analyst",
    description="技术创新专家，能结合玉米生产、加工等相关技术或玉米产成品的替代技术，通过文献研究给出技术发展趋势建议",
    additional_authorized_imports=["time", "numpy", "pandas"],
    max_steps=10,
)


@tool
def manager_planning() -> str:
    """
    This should be called first by the manager to allocate tasks to market_analyst, tech_analyst and manager_agent.
    分配任务
    Returns: market_analyst, tech_analyst和manager_agent各自的任务分配
    """
    with open('plan.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# 创建管理员 Agent
manager_agent = CodeAgent(
    tools=[manager_planning, supply_chain_cases_search, get_supply_chain_analysis_report_format, retriever_tool, final_answer],
    model=model,
    managed_agents=[market_analyst, tech_analyst],
    name="manager_agent",
    description="管理员，首先计划和分配子agent任务，随后负责整合市场分析和技术分析结果，随之提出供应链优化建议，并给出最终报告",
    additional_authorized_imports=["time", "numpy", "pandas"],
    max_steps=10,
)
with open('prompt_templates/supply_prompt.txt', 'r', encoding='utf-8') as file:
    supply_prompt = file.read()
with open('prompt_templates/task_prompt.txt', 'r', encoding='utf-8') as file:
    task_prompt = file.read()

manager_agent.prompt_templates["system_prompt"] += supply_prompt
market_analyst.prompt_templates["managed_agent"]["task"]=task_prompt
tech_analyst.prompt_templates["managed_agent"]["task"]=task_prompt

# print(manager_agent.prompt_templates["system_prompt"])

manager_agent.run("请给出一份2025年-2027年玉米供应链优化建议报告")