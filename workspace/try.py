from smolagents.agents import CodeAgent, ToolCallingAgent
from smolagents import tool, LiteLLMModel
from typing import Optional

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

# 配置 LLM 模型
model = LiteLLMModel(model_id="deepseek/deepseek-chat", api_key="sk-5ca61f10491643eabe3c2de151b8c59e")

# 创建宏观经济学家 Agent
economist_agent = ToolCallingAgent(
    tools=[analyze_energy_impact],
    model=model,
)

economist_agent.run("你是一个宏观经济学家，负责分析国际能源价格对玉米供应链的影响。")