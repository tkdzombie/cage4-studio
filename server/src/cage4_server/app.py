"""Read-only visualization. No simulator, training imports or process control."""
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from cage4_server import reader
from cage4_server.topology import dot_graph

st.set_page_config(page_title='CAGE4 Studio',page_icon='🛡️',layout='wide')
st.title('CAGE4 Studio · 实验观测台')
st.caption('0.3.0 · 只读实验记录 · 算法、训练与评估在独立 PyCharm 研究项目中运行')
roots=[Path(x).expanduser() for x in os.environ.get('CAGE4_RUN_ROOTS','runs').split(os.pathsep) if x]
all_runs=reader.runs(roots)
if not all_runs:
    st.info('尚无实验记录。请在 cage4-baselines 中运行 train 或 evaluate。')
    st.stop()
with st.sidebar:
    selected=st.selectbox('实验记录',all_runs,format_func=lambda p:p.name)
    if st.button('刷新记录'): st.rerun()
    st.caption('数据目录由服务配置。网页不启动或停止训练，也不加载权重。')
    st.caption('TensorBoard 为可选独立服务，可通过 SSH 转发访问本地 6006 端口。')

@st.fragment(run_every='5s')
def monitor():
    state=reader.status(selected)
    a,b,c=st.columns(3)
    a.metric('状态',state.get('state','unknown'))
    b.metric('Episode',state.get('episode',0))
    c.metric('Step',state.get('step',0))
    if state.get('error'): st.error(state['error'])
    if state.get('heartbeat_age_seconds',0)>30:
        st.warning('进程仍存在，但记录超过 30 秒未更新；请检查研究进程日志。')
    data=reader.metrics(selected)
    if not data:
        st.info('等待首个完整 Episode 指标。')
        return
    st.caption('最多显示最近 10,000 个 Episode；完整数据保存在实验目录。')
    st.subheader('Episode Return')
    st.caption('Team Return = 五个智能体的官方 wrapper 回报之和；若官方返回共享团队奖励，此值会重复计数。比较时应保持相同定义。')
    st.line_chart(pd.DataFrame([{'episode':x['episode'],'team_return':x['team_return']} for x in data]).set_index('episode'))
    st.subheader('五个 Blue Agent')
    st.line_chart(pd.DataFrame([{'episode':x['episode'],**x.get('agents',{})} for x in data]).set_index('episode'))
    for key,title in [('actor_loss','Actor loss'),('critic_loss','Critic loss'),('entropy','Entropy')]:
        rows=[{'episode':x['episode'],**{a:v[key] for a,v in x.get('losses',{}).items() if isinstance(v,dict) and key in v}} for x in data]
        if any(len(x)>1 for x in rows):
            st.subheader(title)
            st.line_chart(pd.DataFrame(rows).set_index('episode'))
    st.dataframe(pd.DataFrame(data[-30:]),width='stretch')
monitor()
replay,compare,details=st.tabs(['Episode / Step 回放','对比实验','配置与来源'])
with replay:
    paths=reader.frame_paths(selected)
    episodes=sorted({p.parent.name for p in paths})
    if episodes:
        ep=st.selectbox('Episode',episodes,index=len(episodes)-1)
        options=[p for p in paths if p.parent.name==ep]
        chosen=st.select_slider('已记录 Step',options=options,format_func=lambda p:p.stem,value=options[-1])
        item=reader.read(selected,chosen,{})
        snap=item.get('topology') or (item if 'hosts' in item else None)
        if snap:
            st.caption(snap.get('provenance','旧版拓扑'))
            overview=st.checkbox('仅路由器和互联网主机',value=True)
            subnet=st.selectbox('网段',['全部']+sorted({h['subnet'] for h in snap.get('hosts',[])}))
            st.graphviz_chart(dot_graph(snap,subnet,overview),width='stretch')
            st.dataframe(pd.DataFrame(snap.get('hosts',[])),width='stretch')
        else: st.info('此 Step 保存了动作/观测。完整拓扑按 snapshot_every 间隔保存，请选择带拓扑的 Step。')
        st.subheader('动作、观测与合法性')
        st.caption('Blue legal_mask 仅表示官方固定动作掩码允许；Red not_rejected 表示未记录 InvalidAction，不保证动作成功。')
        st.json({k:v for k,v in item.items() if k!='topology' and k not in ('hosts','links')},expanded=False)
    else: st.info('尚无可回放帧。')
with compare:
    choices=st.multiselect('选择实验',all_runs,default=[selected],format_func=lambda p:p.name)
    curves=[];summaries=[]
    for p in choices:
        rows=reader.metrics(p)
        label=p.parent.name+'/'+p.name
        curves.extend({'episode':x['episode'],'run':label,'return':x['team_return']} for x in rows)
        evaluation=reader.read(p,p/'evaluation.json',{})
        manifest=reader.read(p,p/'manifest.json',{})
        summaries.append({'run':label,'mode':manifest.get('mode','legacy'),
                          'algorithm':manifest.get('algorithm','legacy'),**evaluation})
    if curves:
        st.line_chart(pd.DataFrame(curves).pivot_table(index='episode',columns='run',values='return',aggfunc='last'))
    st.dataframe(pd.DataFrame(summaries),width='stretch')
    st.caption('评估标准差按 seed 均值计算；训练曲线不能作为泛化评估结论。不同 encoder / 步长 / 红队配置的实验需单独解释。')
with details:
    st.json(reader.read(selected,selected/'config.json',{}))
    st.json(reader.read(selected,selected/'manifest.json',{'schema':'legacy'}))
    st.code(str(selected),language=None)
    st.caption('本页 Ground Truth、红队 Session 和 observation 仅供研究者分析；策略模块只接收 Blue observation 特征和官方动作掩码。')
