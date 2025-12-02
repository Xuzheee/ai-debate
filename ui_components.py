import textwrap

# ====================
# CSS 样式常量
# ====================
CUSTOM_CSS = """
<style>
    .agent-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
        margin-bottom: 10px;
        height: 100%;
    }
    .agent-active {
        border: 2px solid #ff4b4b;
        background-color: #fff5f5;
        transform: scale(1.02);
        transition: 0.3s;
    }
    .thought-bubble {
        font-size: 12px;
        color: #555;
        font-style: italic;
        background: #eef;
        padding: 8px;
        border-radius: 5px;
        margin-top: 5px;
        text-align: left;
        border-left: 3px solid #6c5ce7;
    }
    .meta-info {
        font-size: 11px;
        color: #888;
        margin-bottom: 5px;
    }
</style>
"""


# ====================
# HTML 生成函数
# ====================
def generate_card_html(name, conf, state, is_active):
    """
    生成无缩进干扰的 HTML 字符串 (用于 Streamlit 渲染)
    """
    css_class = "agent-card agent-active" if is_active else "agent-card"

    # 限制分数范围
    raw_score = state["score"]
    score = max(0, min(100, raw_score))

    # 计算立场颜色和文字
    if score > 60:
        color = "#ff4b4b"  # 红
        stance = f"🔴 坚定有罪 ({score}%)"
    elif score < 40:
        color = "#4caf50"  # 绿
        stance = f"🟢 倾向无罪 ({score}%)"
    else:
        color = "#f1c40f"  # 黄
        stance = f"🟡 犹豫中 ({score}%)"

    # 获取思维和发言，处理空值
    latest_thought = state["private_memory"][-1] if state["private_memory"] else "..."
    last_speech = state.get("last_speech", "...")

    # 格式化关系网
    relationships = state.get("relationships", {})
    rel_str = ", ".join([f"{k}:{v}" for k, v in relationships.items() if abs(v) > 2])
    if not rel_str:
        rel_str = "中立"

    # 关键：HTML 必须“顶格”写，不能有前导空格，否则在 Markdown 中会被当作代码块而不是 HTML
    card_html = f"""
<div class="{css_class}">
<div style="font-size:28px; margin-bottom:5px;">{conf['avatar']} <b>{name}</b></div>
<div class="meta-info">{conf['occupation']} ({conf['age']}岁)</div>

<div style="margin-top:10px; font-weight:bold; color:{color}; font-size:16px;">
    {stance}
</div>

<div style="width:100%; background-color:#eee; height:8px; border-radius:4px; margin-bottom:10px;">
    <div style="width:{score}%; background-color:{color}; height:8px; border-radius:4px; transition: width 0.5s;"></div>
</div>

<div class="thought-bubble">
    <span style="font-size:14px;">🧠</span> {latest_thought}
</div>

<div style="font-size:14px; margin-top:10px; min-height:60px; font-style: italic; color: #333;">
    🗣️ "{last_speech}"
</div>

<div style="font-size:10px; color:#aaa; margin-top:5px; border-top: 1px solid #eee; padding-top:5px;">
    ❤️ 关系: {rel_str}
</div>
</div>
"""
    return card_html