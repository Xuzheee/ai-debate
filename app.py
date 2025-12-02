import streamlit as st
import json
import re
import time
import random
from typing import Dict, List

# --- LangChain 新增引入 ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field

# --- 你的自定义模块 ---
import case_config as config
import ui_components  # 导入 UI 组件库

# ====================
# 1. 基础配置
# ====================
st.set_page_config(page_title="六怒汉：深度个体模拟", layout="wide")

# 加载来自 ui_components 的 CSS 样式
st.markdown(ui_components.CUSTOM_CSS, unsafe_allow_html=True)

if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "DeepSeek"
if "model_name" not in st.session_state:
    st.session_state.model_name = "deepseek-chat"
if "api_base" not in st.session_state:
    st.session_state.api_base = ""

# --- 侧边栏控制台 (保持你原有的逻辑不变) ---
with st.sidebar:
    st.header("🧠 上帝视角控制台")

    # 模型提供商选择
    st.session_state.llm_provider = st.selectbox(
        "选择大模型提供商",
        options=["DeepSeek", "OpenAI", "自定义(OpenAI兼容)"],
        index=["DeepSeek", "OpenAI", "自定义(OpenAI兼容)"].index(
            st.session_state.llm_provider
        )
        if st.session_state.llm_provider in ["DeepSeek", "OpenAI", "自定义(OpenAI兼容)"]
        else 0,
    )

    # 根据提供商设置默认模型名与提示文案
    if st.session_state.llm_provider == "DeepSeek":
        key_label = "DeepSeek API Key"
        default_model = "deepseek-chat"
    elif st.session_state.llm_provider == "OpenAI":
        key_label = "OpenAI API Key"
        default_model = "gpt-4o-mini"
    else:  # 自定义 OpenAI 兼容
        key_label = "API Key"
        default_model = st.session_state.model_name or "your-model-name"

    st.session_state.api_key = st.text_input(
        key_label, type="password", value=st.session_state.api_key
    )

    st.session_state.model_name = st.text_input(
        "模型名称",
        value=st.session_state.model_name or default_model,
        help="例如：DeepSeek 使用 deepseek-chat；OpenAI 使用 gpt-4o / gpt-4o-mini 等",
    )

    # 只有在“自定义(OpenAI兼容)”模式下才需要填写 base_url
    if st.session_state.llm_provider == "自定义(OpenAI兼容)":
        st.session_state.api_base = st.text_input(
            "自定义 API Base URL",
            value=st.session_state.api_base or "",
            placeholder="例如：https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    if not st.session_state.api_key: st.warning("请输入 Key"); st.stop()
    
    st.divider()
    auto_rounds = st.number_input("自动运行轮数", 1, 15, 1)
    run_btn = st.button("▶️ 开始深度模拟", type="primary")
    
    st.divider()
    if st.button("🗑️ 重置世界"): st.session_state.clear(); st.rerun()

# 初始化 LLM (保持不变)
if st.session_state.llm_provider == "DeepSeek":
    llm = ChatOpenAI(
        model=st.session_state.model_name or "deepseek-chat",
        openai_api_key=st.session_state.api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.9,
    )
elif st.session_state.llm_provider == "OpenAI":
    llm = ChatOpenAI(
        model=st.session_state.model_name or "gpt-4o-mini",
        openai_api_key=st.session_state.api_key,
        temperature=0.9,
    )
else:
    llm = ChatOpenAI(
        model=st.session_state.model_name,
        openai_api_key=st.session_state.api_key,
        openai_api_base=st.session_state.api_base or None,
        temperature=0.9,
    )

# ====================
# 2. 状态初始化 (升级版：独立记忆)
# ====================
if "history" not in st.session_state:
    # 这是全局公开的剧本，用于UI显示
    st.session_state.history = [{"role": "Foreman", "content": "第一轮投票 5:1。Davis，请陈述你的理由。"}]

if "agents_memories" not in st.session_state:
    st.session_state.agents_memories = {}
    
    for name in config.AGENTS:
        # 不同人可以有不同的记忆力 (k值)
        # 例如：老年人(McCardle)记忆短，建筑师(Davis)记忆长
        k_value = 5 if config.AGENTS[name]['age'] > 65 else 10
        
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=k_value,
            return_messages=True
        )
        
        # 预埋初始背景
        memory.chat_memory.add_user_message("System: 案件审理开始。请基于你的证据和直觉进行辩论。")
        st.session_state.agents_memories[name] = memory

    if "agents_state" not in st.session_state:
        st.session_state.agents_state = {}
        for name in config.AGENTS:
            st.session_state.agents_state[name] = {
                "score": config.AGENTS[name]["init_score"],
                "last_speech": "...",
                # ↓↓↓ 注意这里，大括号后面必须加逗号
                "relationships": {other: 0 for other in config.AGENTS if other != name}, 
                
                # 注意：private_memory 现在可以由 LangChain Memory 接管一部分，
                # 但为了保留“长期深层记忆”，我们依然保留这个列表
                "private_memory": config.AGENTS[name].get("initial_memory", []), 
            }

if "current_speaker" not in st.session_state: st.session_state.current_speaker = None


# ====================
# 3. LangChain 结构定义 (升级版)
# ====================

class JurorAction(BaseModel):
    internal_thought: str = Field(description="内心独白：分析证据，评价上一位发言者，不要说客套话。")
    
    # 新增：公开立场，用于UI显示标签，而不是显示数字
    public_stance: str = Field(description="公开表达的立场，只能是以下三个之一：['无罪', '犹豫', '有罪']")
    
    # 关键修改：明确要求发言中不要带数字
    speech: str = Field(description="公开发言：用自然的口语表达，严禁在话语中直接说出分数数值！比如不要说'我打80分'，要说'我非常确信他有罪'。")
    
    relationship_update: Dict[str, int] = Field(description="好感度变化：{'人名': -5到5之间的整数}")
    
    # 这个分数依然保留，作为底层驱动，但对其他Agent不可见
    new_score: int = Field(description="内心真实的定罪确信度 (0-100)。0=确信无罪，100=确信有罪。")

juror_parser = PydanticOutputParser(pydantic_object=JurorAction)

agent_template_str = """
你现在是: {name} (年龄: {age}, 职业: {occupation})。

【人物设定】:
{backstory}
【核心价值观】: {core_values}

【当前状态】:
你的内心定罪分数: {current_score}/100。
局势感知(上一位发言者): {last_speaker_name} (好感度: {last_speaker_rel})

【你的专属记忆流】:
以下是你脑海中关于最近对话的记忆（Human代表其他人，AI代表你自己）：
{chat_history}

【思考任务】:
1. 回顾【记忆流】，注意你之前的立场和你对他人的看法。
2. 结合上一句发言进行回应。

【输出要求】:
(保持原有 JSON 格式要求)
{format_instructions}
"""

agent_prompt_template = PromptTemplate(
    template=agent_template_str,
    input_variables=[
        "name", "age", "occupation", "backstory", "core_values", 
        "current_score", "last_speaker_name", "last_speaker_rel", 
        "chat_history" # <--- 变量名变了
    ],
    partial_variables={"format_instructions": juror_parser.get_format_instructions()}
)

agent_prompt_template = PromptTemplate(
    template=agent_template_str,
    input_variables=[
        "name", "age", "occupation", "backstory", "core_values", 
        "speaking_style", "current_score", "private_memory", 
        "last_speaker_name", "last_speaker_rel", "case_background", "history_text"
    ],
    partial_variables={"format_instructions": juror_parser.get_format_instructions()}
)


# ====================
# 4. 核心逻辑 (整合了你的逻辑和LangChain)
# ====================

def run_one_turn():
    # --- A. 选人逻辑 (保持不变) ---
    visible_history = st.session_state.history[-10:]
    history_text_for_supervisor = "\n".join([f"{m['role']}: {m['content']}" for m in visible_history])
    
    recent_speakers = [m["role"] for m in st.session_state.history[-3:]]
    candidates = [n for n in config.AGENTS if n not in recent_speakers]
    if not candidates:
        last = recent_speakers[-1] if recent_speakers else ""
        candidates = [n for n in config.AGENTS if n != last]

    try:
        # (Supervisor 代码省略，保持原样) ...
        # 假设这里选出了 next_speaker
        next_speaker = random.choice(candidates) # 或者你原来的逻辑
        st.session_state.current_speaker = next_speaker

        # --- B. 深度模拟逻辑 (Memory 接入) ---
        state = st.session_state.agents_state[next_speaker]
        conf = config.AGENTS[next_speaker]
        
        # 1. 获取该 Agent 的独立记忆对象
        agent_memory = st.session_state.agents_memories[next_speaker]
        
        # 2. 从 Memory 中加载历史记录 (格式化为字符串)
        # load_memory_variables 返回的是一个字典，包含 chat_history
        memory_vars = agent_memory.load_memory_variables({})
        chat_history_str = str(memory_vars.get("chat_history", ""))

        last_msg = st.session_state.history[-1]
        last_speaker = last_msg["role"]
        rel_score = state["relationships"].get(last_speaker, 0)
        current_score = state["score"]

        # 3. 构造 Prompt
        final_prompt = agent_prompt_template.format(
            name=next_speaker,
            age=conf['age'],
            occupation=conf['occupation'],
            backstory=conf['backstory'],
            core_values=conf['core_values'],
            current_score=current_score,
            last_speaker_name=last_speaker,
            last_speaker_rel=rel_score,
            chat_history=chat_history_str  # <--- 传入独立记忆字符串
        )

        response = llm.invoke([HumanMessage(content=final_prompt)])
        parsed_action = juror_parser.parse(response.content)
        data = parsed_action.dict()

        if data:
            # --- C. 处理惯性和状态 (保持不变) ---
            target_score = int(data.get("new_score", current_score))
            delta = target_score - current_score
            max_change = 15
            if delta > max_change: delta = max_change
            elif delta < -max_change: delta = -max_change
            real_new_score = max(0, min(100, current_score + delta))
            
            state["score"] = real_new_score
            state["last_speech"] = data["speech"]
            
            # --- 🔥 D. 关键：广播更新所有人的记忆 ---
            speech_content = data["speech"]
            internal_thought = data["internal_thought"]

            for agent_name, mem in st.session_state.agents_memories.items():
                if agent_name == next_speaker:
                    # 对于【我自己】：
                    # 我要把“内心独白”+“公开讲话”都存进去，形成 Chain of Thought
                    # 这样我下次就能记得我为什么这么说了
                    combined_input = f"(内心独白: {internal_thought}) -> 我说: {speech_content}"
                    mem.chat_memory.add_ai_message(combined_input)
                else:
                    # 对于【其他人】：
                    # 他们只能听到我的“公开讲话”
                    mem.chat_memory.add_user_message(f"{next_speaker} 说: {speech_content}")

            # 处理关系更新 (保持不变)
            for target, change in data["relationship_update"].items():
                if target in state["relationships"]:
                    state["relationships"][target] += int(change)

            # 写入全局 UI 历史
            st.session_state.history.append({
                "role": next_speaker, 
                "content": speech_content,
                "stance": data["public_stance"]
            })
            return True

        return False

    except Exception as e:
        st.error(f"Error: {e}")
        return False



# ====================
# 5. 界面渲染 (保持不变)
# ====================
st.title("⚖️ 十二怒汉：深度个体模拟")

st.subheader("🏛️ 陪审团席位 (上帝视角)")
cols = st.columns(3)

# 渲染卡片
for i, name in enumerate(config.AGENTS.keys()):
    state = st.session_state.agents_state[name]
    conf = config.AGENTS[name]
    is_active = (name == st.session_state.current_speaker)

    with cols[i % 3]:
        html_code = ui_components.generate_card_html(name, conf, state, is_active)
        st.markdown(html_code, unsafe_allow_html=True)

# --- 自动运行循环 ---
if run_btn:
    bar = st.progress(0)
    for i in range(auto_rounds):
        success = run_one_turn()
        bar.progress((i+1)/auto_rounds)
        if not success: break
        time.sleep(1) 
    st.rerun()

# --- 历史记录 ---
# --- 历史记录 ---
st.divider()
st.subheader("📜 案件记录")
for msg in reversed(st.session_state.history):
    role = msg['role']
    avatar = config.AGENTS.get(role, {}).get("avatar", "🤖")
    
    if role == "System":
        st.info(msg['content'])
    else:
        with st.chat_message(role, avatar=avatar):
            # 获取立场标签，如果没有（旧记录）则不显示
            stance = msg.get("stance", "")
            
            # 定义标签颜色
            badge_color = "gray"
            if stance == "有罪": badge_color = "red"
            elif stance == "无罪": badge_color = "green"
            elif stance == "犹豫": badge_color = "orange"
            
            # 显示内容：如果存在立场，先显示立场徽章
            if stance:
                st.markdown(f":{badge_color}[【{stance}】] {msg['content']}")
            else:
                st.write(msg['content'])
            
            # (可选) 如果你是调试模式，可以把这一行取消注释看看真实分数变化
            # st.caption(f"Debug Score: {msg.get('score_debug', 'N/A')}")
