from smolagents import tool, LiteLLMModel, HfApiModel, CodeAgent
from smolagents import DuckDuckGoSearchTool, GoogleSearchTool, VisitWebpageTool
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
def analyze_crude_oil_market(year: int) -> str:
    """
    分析国际原油市场的供需趋势和价格走势。
    Args:
        year: 分析的年份
    Returns:
        原油市场分析报告
        任何涉及数据的信息必须输出其来源
    """

    return f"{year}年国际原油市场分析报告："

@tool
def analyze_fuel_ethanol_market(year: int) -> str:
    """
    分析国际燃料乙醇市场的供需趋势和价格走势。
    Args:
        year: 分析的年份
    Returns:
        燃料乙醇市场分析报告
        任何涉及数据的信息必须输出其来源
    """

    return f"{year}年国际燃料乙醇市场分析报告："

@tool
def predict_corn_market(year: int, scenario: str) -> str:
    """
    预测国际和国内玉米市场的生产量、播种面积和价格走势。
    Args:
        year: 预测的年份
        scenario: 情景分析（基准、乐观、悲观）
    Returns:
        玉米市场预测报告
        任何涉及数据的信息必须输出其来源
    """
  
    return f"{year}年玉米市场预测报告（{scenario}情景）："

@tool
def analyze_seed_technology(year: int) -> str:
    """
    分析玉米种子技术的发展趋势。
    Args:
        year: 分析的年份
    Returns:
        种子技术分析报告
        必须输出其来源
    """

    return f"{year}年玉米种子技术分析报告："

@tool
def analyze_production_technology(year: int) -> str:
    """
    分析玉米生产技术（如滴灌）的发展趋势。
    Args:
        year: 分析的年份
    Returns:
        生产技术分析报告
        必须输出其来源
    """
    return f"{year}年玉米生产技术分析报告："

@tool
def analyze_processing_technology(year: int) -> str:
    """
    分析玉米加工技术的发展趋势。
    Args:
        year: 分析的年份
    Returns:
        加工技术分析报告
        必须输出其来源
    """
    return f"{year}年玉米加工技术分析报告："


# The default access is api.coze.com, but if you need to access api.coze.cn,
# please use base_url to configure the api endpoint to access
# coze_api_base = os.getenv("COZE_API_BASE") or COZE_COM_BASE_URL

# Init the Coze client through the access_token.


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
    market_analysis = market_analyst.predict_corn_market(2025, scenario)
    tech_analysis = tech_analyst.analyze_seed_technology(2025)
    return f"供应链优化建议报告（{scenario}情景）：\n市场分析：{market_analysis}\n技术分析：{tech_analysis}"

@tool
def search_successful_cases() -> str:
    """
    搜索国际大型粮食企业的成功案例。
    Returns:
        成功案例报告
    """
    return f"国际大型粮食企业成功案例报告："

search_tool = DuckDuckGoSearchTool()
# google_tool = GoogleSearchTool() # need to set SERPAPI_API_KEY
# visitweb_tool = VisitWebpageTool()

# 创建市场分析专家 Agent
market_analyst = CodeAgent(
    tools=[analyze_crude_oil_market, analyze_fuel_ethanol_market, predict_corn_market, search_tool],
    model=model,
    name="market_analyst",
    description="市场分析专家，能结合特朗普上一任期的历史政策、原油和燃料乙醇市场的价格历史数据对玉米供需关系进行预测",
)

# 创建技术创新专家 Agent
"""tech_analyst = CodeAgent(
    tools=[analyze_seed_technology, analyze_production_technology, analyze_processing_technology, search_tool],
    model=model,
    name="tech_analyst",
    description="技术创新专家，能结合玉米生产、加工等相关技术或玉米产成品的替代技术，通过文献研究给出技术发展趋势建议",
)"""
tech_analyst = CodeAgent(
    tools=[tech_search],
    model=model,
    name="tech_analyst",
    description="技术创新专家，能结合玉米生产、加工等相关技术或玉米产成品的替代技术，通过文献研究给出技术发展趋势建议",
)


@tool
def manager_planning() -> str:
    """
    分配任务
    Returns: market_analyst, tech_analyst和manager_agent各自的任务分配
    """
    with open('plan.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# 创建管理员 Agent
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[tech_analyst, manager_planning],
    name="manager_agent",
    description="管理员，负责整合技术分析结果，提出供应链优化建议，给出最终报告",
    additional_authorized_imports=["time", "numpy", "pandas"],
)

manager_agent.run("请给出一份2025年-2027年玉米供应链优化建议报告")