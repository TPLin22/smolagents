from smolagents import tool, LiteLLMModel, HfApiModel, CodeAgent
from smolagents import DuckDuckGoSearchTool, GoogleSearchTool, VisitWebpageTool
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

HF_Token = os.getenv("HUGGINGFACE_TOKEN")
ds_api_token = os.getenv("DEEPSEEK_API_TOKEN")

# 配置 LLM 模型

# model is "Qwen/Qwen2.5-Coder-32B-Instruct" by default, can be set by model_id
# model = HfApiModel(token=HF_Token) 
model = LiteLLMModel(model_id="deepseek/deepseek-chat", api_key=ds_api_token)


@tool
def analyze_energy_impact(energy_price: float) -> str:
    """
    分析国际能源价格对玉米供应链的宏观经济影响。
    Args:
        energy_price: 国际能源价格
    Returns:
        分析报告
    """
    return f"能源价格为 {energy_price} 时，玉米供应链的宏观经济影响分析报告..."

search_tool = DuckDuckGoSearchTool()
google_tool = GoogleSearchTool() # need to set SERPAPI_API_KEY
visitweb_tool = VisitWebpageTool()

# 创建市场分析专家 Agent
market_analyst = CodeAgent(
    tools=[visitweb_tool, search_tool],
    model=model,
    description="市场分析专家，能结合历史政策、能源价格和玉米供需关系进行预测",
)

# 创建技术创新专家 Agent
tech_analyst = CodeAgent(
    tools=[visitweb_tool, search_tool],
    model=model,
    description="技术创新专家，能结合玉米生产、加工等相关技术或玉米产成品的替代技术，通过文献研究给出技术发展趋势建议",
)

manager_agent = CodeAgent(
    tools=[], model=model, managed_agents=[market_analyst, tech_analyst]
)

manager_agent.run("川普在上一任期采取了怎样的能源、粮食政策？结合2023-2024年的乙醇市场变化，预测2025年国内粮食市场的价格变化趋势。")