from smolagents import tool, LiteLLMModel, HfApiModel, CodeAgent
from smolagents import DuckDuckGoSearchTool, GoogleSearchTool, VisitWebpageTool
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
# model = HfApiModel(token=HF_Token) 
model = LiteLLMModel(model_id="deepseek/deepseek-chat", api_key=ds_api_token)

@tool
def get_market_analysis_report_format() -> str:
    """
    获取市场分析报告的格式模板。
    Returns:
        市场分析报告的格式模板
    """
    try:
        with open('market_analysis_report_format.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "未找到市场分析报告格式文件，请确保文件 'market_analysis_report_format.txt' 存在。"
    except Exception as e:
        return f"读取市场分析报告格式文件时发生错误：{str(e)}"
    

# The default access is api.coze.com, but if you need to access api.coze.cn,
# please use base_url to configure the api endpoint to access
# coze_api_base = os.getenv("COZE_API_BASE") or COZE_COM_BASE_URL

# Init the Coze client through the access_token.
@tool
def market_data_search(question: str) -> str:
    """
    根据提问，搜索市场相关数据（如价格、供需趋势等）。
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
def optimize_supply_chain(scenario: str) -> str:
    """
    根据市场和技术分析结果，提出供应链优化建议。
    Args:
        scenario: 情景分析（基准、乐观、悲观）
    Returns:
        供应链优化建议报告
    """
    return f"供应链优化建议报告（{scenario}情景）：\n市场分析：{market_analysis}\n技术分析：{tech_analysis}"

@tool
def search_successful_cases() -> str:
    """
    搜索国际大型粮食企业的成功案例。
    Returns:
        成功案例报告
    """
    return f"国际大型粮食企业成功案例报告："



# 创建市场分析专家 Agent
market_analyst = CodeAgent(
    tools=[market_data_search, policy_search, get_market_analysis_report_format],
    model=model,
    name="market_analyst",
    description="市场分析专家，通过2023-2024年的国际原油市场与国际燃料乙醇市场的变化（价格与供需趋势），结合川普上台后所采取或可能采取的能源发展政策，并参考2017年-2020年川普上台后的历史政策、国际能源市场、国际粮食市场的既往趋势，给出预期2025年-2027年国际国内粮食市场有关的趋势分析（如生产量、播种面积、价格走势），重点给出我国玉米市场生产与贸易的相关分析结果。",
    additional_authorized_imports=["time", "numpy", "pandas"],
)


tech_analyst = CodeAgent(
    tools=[tech_search],
    model=model,
    name="tech_analyst",
    description="技术创新专家，能结合玉米生产、加工等相关技术或玉米产成品的替代技术，通过文献研究给出技术发展趋势建议",
    additional_authorized_imports=["time", "numpy", "pandas"],
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

CHINESE_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="You are a helpful analyst. You should give your answers in Chinese.",
    planning=PlanningPromptTemplate(
        initial_facts="",
        initial_plan="",
        update_facts_pre_messages="",
        update_facts_post_messages="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=manager_planning()),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)

# 创建管理员 Agent
manager_agent = CodeAgent(
    tools=[manager_planning],
    model=model,
    managed_agents=[tech_analyst, market_analyst],
    name="manager_agent",
    description="管理员，首先计划和分配子agent任务，随后负责整合市场分析和技术分析结果，提出供应链优化建议，给出最终报告",
    prompt_templates=CHINESE_PROMPT_TEMPLATES,
    #report=manager_planning(),
    additional_authorized_imports=["time", "numpy", "pandas"],
)

manager_agent.run("请给出一份2025年-2027年玉米供应链优化建议报告")