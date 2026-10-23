"""Read-only CAGE4 experiment view."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import altair as alt
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='CAGE4 · 算法实验台',layout='wide')
st.title('CAGE4 · 算法实验台')
st.caption('一个训练 seed · 官方原始团队回报 · 完整任务；不同算法/协议分开记录')
@st.fragment(run_every='10s')
def view():
    runs=sorted((ROOT/'runs').glob('*/metrics.jsonl'), key=lambda p:p.stat().st_mtime,reverse=True)
    if not runs:st.info('尚无完成的训练更新');return
    selected=st.selectbox('实验', [str(p.parent.name) for p in runs])
    path=next(p for p in runs if p.parent.name==selected);rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    if not rows:return
    data=pd.DataFrame([{k:v for k,v in r.items() if k!='learner'} for r in rows])
    cols=st.columns(3)
    cols[0].metric('最近团队回报',f"{rows[-1]['return']:.1f}")
    cols[1].metric('仿真交互步数',f"{rows[-1]['env_steps']:,}")
    cols[2].metric('完成更新',len(rows))
    data['最近5轮均值']=data['return'].rolling(5).mean()
    chart=data.rename(columns={'return':'每轮原始回报'}).melt(id_vars=['env_steps'],value_vars=['每轮原始回报','最近5轮均值'],var_name='曲线',value_name='团队回报')
    st.altair_chart(alt.Chart(chart).mark_line().encode(x=alt.X('env_steps:Q',title='仿真交互步数'),y=alt.Y('团队回报:Q',scale=alt.Scale(zero=False)),color='曲线:N',tooltip=['env_steps','曲线','团队回报']),use_container_width=True)
    conf=json.loads((path.parent/'config.json').read_text())
    protocol=conf.get('protocol','repaired')
    if conf.get('level')=='flat':
        st.write(f"算法：{conf.get('algorithm','ppo')} · 五个直接动作策略端到端训练；"+('使用公开作者权重初始化，所有网络继续优化。' if conf.get('warm_start') else '全部随机初始化，没有预训练或固定子策略。'))
    else:st.write('作者15个子策略冻结，训练5个Master；不能当作完整端到端重训。')
    st.caption(f"协议：{protocol} · learner reward单位：{conf.get('reward_scale',1)} · PPO λ：{conf.get('gae_lambda',.95)}；环境及展示回报不缩放")
    summary=path.parent/'summary.json'
    if summary.exists():
        obj=json.loads(summary.read_text());st.write(f"已停止：{obj['reason']}。单看训练横盘不能证明收敛。")
    else:st.info('训练中，等待真实收益与平台证据。')
    st.subheader('同一协议、共同探索随机数的对照')
    ref=[]
    names=sorted(p.name for p in ROOT.glob('*comparison*.json'))
    for fname in names:
        p=ROOT/fname
        if not p.exists():continue
        obj=json.loads(p.read_text())
        if obj.get('protocol')!=protocol or (obj.get('run') and obj.get('run')!=selected):continue
        for r in obj['rows']:
            ref.append({'对照':fname,'策略':r['endpoint'],'场景seed':r['seed'],'团队回报':r['return'],'仿真步数':r['ticks']})
    if ref:st.dataframe(pd.DataFrame(ref),hide_index=True,use_container_width=True)
    else:st.info('尚无此协议的起终点对照。')
    with st.expander('原始训练数据与诊断'):
        st.dataframe(data,hide_index=True,use_container_width=True)
        st.json(rows[-1].get('learner',{}))
    with st.expander('协议与证据边界'):
        st.write('官方500配置步，实际499 transition；奖励不修改，五名Blue同一团队回报只展示一次。动作mask与官方合法槽相交；多步动作期间只允许Sleep；保留现有特征供critic使用；有限任务终点不bootstrap。history-correct在Remove后保留已有IOC，成功Restore后才重置。')
        st.write('沿用作者从state提取事件、阻断和decoy元数据的观测接口，不冒称纯Blue observation；未加入红队session/感染真值。当前平坦策略从原始动作端到端训练。作者原Flat权重默认关闭流量控制；在全动作协议上的表现属于迁移参考，不能当作原论文性能复现。两个场景用于快速诊断；小规模新场景对照也不代表大范围泛化结论。')
view()
