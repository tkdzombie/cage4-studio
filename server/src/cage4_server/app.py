"""Read-only visualization. No simulator, training imports or process control."""
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from cage4_server import reader
from cage4_server.topology import dot_graph

st.set_page_config(page_title='CAGE4 Studio',page_icon='🛡️',layout='wide')
st.title('CAGE4 Studio · 实验观测台')
st.caption('只读实验记录 · 每 5 秒刷新 · 训练与评估在独立研究进程中运行')
roots=[Path(x).expanduser() for x in os.environ.get('CAGE4_RUN_ROOTS','runs').split(os.pathsep) if x]
SHOW_HISTORY=os.environ.get('CAGE4_SHOW_HISTORY','0') == '1'
all_runs=reader.runs(roots)
if not SHOW_HISTORY: all_runs=all_runs[:1]
if not all_runs:
    st.info('尚无实验记录。请在 cage4-baselines 中运行 train 或 evaluate。')
    st.stop()
with st.sidebar:
    if SHOW_HISTORY:
        selected=st.selectbox('实验记录',all_runs,format_func=lambda p:p.name)
    else:
        selected=all_runs[0]
        st.caption('当前实验：'+selected.name)
        st.caption('新实验开始后自动切换；历史记录不在本页显示。')
    if st.button('刷新记录'): st.rerun()
    st.caption('数据目录由服务配置。网页不启动或停止训练，也不加载权重。')
    st.caption('TensorBoard 为可选独立服务，可通过 SSH 转发访问本地 6006 端口。')

@st.fragment(run_every='5s')
def monitor():
    if not SHOW_HISTORY:
        latest=reader.runs(roots)
        if latest and latest[0] != selected: st.rerun()
    state=reader.status(selected)
    data=reader.metrics(selected)
    metadata=reader.read(selected,selected/'manifest.json',{})
    is_ot=metadata.get('schema')==reader.OT_SCHEMA
    if state.get('state') == 'finished' and data:
        state={**state,'episode':data[-1]['episode'],'step':data[-1]['step']}
    a,b,c=st.columns(3)
    a.metric('状态',state.get('state','unknown'))
    b.metric('已完成 Episode',state.get('completed_episodes',state.get('episode',0)))
    c.metric('Step',state.get('step',0))
    if is_ot:
        config=metadata.get('config',{})
        goal=config.get('episodes',0)
        completed=state.get('completed_episodes',0)
        st.progress(min(completed/max(goal,1),1),text=f'训练 {completed} / {goal} 回合')
        st.caption(f"算法：{metadata.get('variant','MACPO-OT')}；训练 seed：{config.get('seed')}；阶段：{'训练' if state.get('phase')=='collecting' else '开发评估'}；每 {config.get('rollout_episodes')} 回合更新。")
    execution=reader.read(selected,selected/'manifest.json',{}).get('execution',{})
    if execution:
        d,e,f=st.columns(3)
        d.metric('并行环境',state.get('num_envs',execution.get('num_envs',1)))
        speed=state.get('transitions_per_second')
        e.metric('总体采样速度',f'{speed:.1f} 环境步/秒' if isinstance(speed,(int,float)) else '测量中')
        phases={'collecting':'并行采样','updating':'更新模型','recording':'写入指标'}
        f.metric('阶段',phases.get(state.get('phase'),state.get('phase','初始化')))
        st.caption(f"模型计算线程：{execution.get('torch_threads',1)}；速度包含初始化、采样、更新和保存开销。")
    if state.get('error'): st.error(state['error'])
    if state.get('state')=='running' and state.get('heartbeat_age_seconds',0)>30:
        st.warning('进程仍存在，但记录超过 30 秒未更新；请检查研究进程日志。')
    if not data:
        st.info('等待首个完整 Episode 指标。')
        return
    st.caption('最多显示最近 10,000 个 Episode；完整数据保存在实验目录。')
    st.subheader('Episode Return')
    metric = 'official_team_return' if all('official_team_return' in x for x in data) else 'team_return'
    st.caption('官方团队回报：共享 reward 每步计一次，包含已执行的终止步。' if metric == 'official_team_return' else
               '旧版 team_return：五个 Blue 回报之和，可能重复共享奖励；与官方团队回报分开比较。')
    returns=pd.DataFrame([{'episode':x['episode'],metric:x[metric]} for x in data]).set_index('episode')
    returns['50回合滑动平均']=returns[metric].rolling(50,min_periods=10).mean()
    st.line_chart(returns)
    if is_ot:
        st.subheader('OT 连续性约束')
        st.caption('任一受保护 OT 服务连续不可用到第 17 步，整回合事件记 1。训练事件率是样本统计，不是安全认证。')
        risk=pd.DataFrame([{'episode':x['episode'],'回合事件（0/1）':x['ot_event'],'风险预算':x['event_budget']} for x in data]).set_index('episode')
        risk['最近 50 回合事件率']=risk['回合事件（0/1）'].rolling(50,min_periods=1).mean()
        st.line_chart(risk)
        st.caption('当前训练不保存逐步完整回放，以控制磁盘和运行开销。下方显示真实更新诊断。')
        for key,title in [('actual_policy_kl','实际策略 KL'),('reward_mc_mse_after','Reward critic MC MSE（更新后）'),('cost_mse_after','Cost critic MSE（更新后）')]:
            rows=[{'episode':x['episode'],**{a:v[key] for a,v in x['losses'].items() if key in v}} for x in data if x['losses']]
            if rows:
                st.subheader(title)
                st.line_chart(pd.DataFrame(rows).set_index('episode'))
        evaluations=reader.read(selected,selected/'evaluations.json',[])
        if evaluations:
            st.subheader('开发评估（随机主模式 / greedy 辅助）')
            st.dataframe(pd.DataFrame([{k:v for k,v in x.items() if k!='ot'} for x in evaluations]),width='stretch')
    safety = [{'episode':x['episode'], '业务失败率': x['safety']['mean_cost'],
               '预算': x['safety_budget_rate']} for x in data if 'safety' in x]
    if safety:
        st.subheader('业务可用性约束')
        st.caption('越低越好；按官方任务阶段权重统计 Green 工作/服务失败。预算属于研究协议。')
        st.line_chart(pd.DataFrame(safety).set_index('episode'))
    if not is_ot:
        st.subheader('五个 Blue Agent')
        st.line_chart(pd.DataFrame([{'episode':x['episode'],**x.get('agents',{})} for x in data]).set_index('episode'))
    for key,title in [('actor_loss','Actor loss'),('critic_loss','Critic loss'),('entropy','Entropy')]:
        rows=[{'episode':x['episode'],**{a:v[key] for a,v in x.get('losses',{}).items() if isinstance(v,dict) and key in v}} for x in data]
        if any(len(x)>1 for x in rows):
            st.subheader(title)
            st.line_chart(pd.DataFrame(rows).set_index('episode'))
    st.dataframe(pd.DataFrame(data[-30:]),width='stretch')
monitor()
if SHOW_HISTORY:
    replay,compare,details=st.tabs(['Episode / Step 回放','对比实验','配置与来源'])
else:
    replay,details=st.tabs(['Episode / Step 回放','配置与来源'])
    compare=None
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
if compare is not None:
    with compare:
        choices=st.multiselect('选择实验',all_runs,default=[selected],format_func=lambda p:p.name)
        metric = st.selectbox('回报口径', ['official_team_return', 'team_return'])
        st.caption('仅显示具有所选指标的记录，避免新旧回报定义混用。')
        curves=[];summaries=[]
        for p in choices:
            rows=reader.metrics(p)
            label=p.parent.name+'/'+p.name
            curves.extend({'episode':x['episode'],'run':label,'return':x[metric]} for x in rows if metric in x)
            evaluation=reader.read(p,p/'evaluation.json',{})
            manifest=reader.read(p,p/'manifest.json',{})
            summaries.append({'run':label,'mode':manifest.get('mode','legacy'),
                              'algorithm':manifest.get('algorithm','legacy'),**evaluation})
        if curves:
            st.line_chart(pd.DataFrame(curves).pivot_table(index='episode',columns='run',values='return',aggfunc='last'))
        st.dataframe(pd.DataFrame(summaries),width='stretch')
        st.caption('评估标准差按 seed 均值计算；训练曲线不能作为泛化评估结论。不同 encoder / 步长 / 红队配置的实验需单独解释。')
with details:
    st.json(reader.read(selected,selected/'config.json',reader.read(selected,selected/'manifest.json',{}).get('config',{})))
    st.json(reader.read(selected,selected/'manifest.json',{'schema':'legacy'}))
    st.code(str(selected),language=None)
    st.caption('本页 Ground Truth、红队 Session 和 observation 仅供研究者分析；策略模块只接收 Blue observation 特征和官方动作掩码。')
