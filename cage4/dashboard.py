"""Read-only CAGE4 experiment view."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import altair as alt
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='CAGE4 · 算法实验台',layout='wide')
st.title('CAGE4 · 算法实验台')
st.caption('一个训练 seed · 官方原始团队回报 · 完整任务；不同算法/协议分开记录')
@st.fragment(run_every='10s')
def view():
    runs=sorted((ROOT/'runs').glob('*/metrics.jsonl'), key=lambda p:p.stat().st_mtime,reverse=True)
    if not runs:st.info('尚无实验。配置 configs/ppo.json 或 configs/my_algorithm.json 后运行 run.py。');return
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
    st.altair_chart(alt.Chart(chart).mark_line().encode(x=alt.X('env_steps:Q',title='仿真交互步数'),y=alt.Y('团队回报:Q',scale=alt.Scale(zero=False)),color='曲线:N',tooltip=['env_steps','曲线','团队回报']),width='stretch')
    conf=json.loads((path.parent/'config.json').read_text())
    protocol=conf.get('protocol','repaired')
    initialization=conf.get('initialization','author' if conf.get('warm_start') else 'scratch')
    st.write(f"算法：{conf.get('algorithm','ppo')} · 五个直接动作策略端到端训练；"+('作者权重初始化，所有网络继续优化。' if initialization=='author' else '随机初始化，没有预训练或冻结子策略。'))
    st.caption(f"协议：{protocol} · learner reward单位：{conf.get('reward_scale',1)} · PPO λ：{conf.get('gae_lambda',.95)}；环境及展示回报不缩放")
    summary=path.parent/'summary.json'
    if summary.exists():
        obj=json.loads(summary.read_text());st.write(f"已停止：{obj['reason']}。单看训练横盘不能证明收敛。")
    else:
        status_path=path.parent/'status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        st.info(f"阶段：{status.get('phase','training')} · 预算 {conf.get('iterations')} 次更新 · 随机种子 {conf.get('seed')}")
    st.subheader('同一协议、共同探索随机数的对照')
    ref=[]
    names=[path.parent/'evaluation.json']
    for fname in names:
        p=Path(fname)
        if not p.exists():continue
        obj=json.loads(p.read_text())
        if obj.get('protocol')!=protocol or (obj.get('run') and obj.get('run')!=selected):continue
        for r in obj['rows']:
            ref.append({'对照':p.name,'策略':r['endpoint'],'场景seed':r['seed'],'团队回报':r['return'],'仿真步数':r['ticks']})
    if ref:st.dataframe(pd.DataFrame(ref),hide_index=True,width='stretch')
    else:st.info('尚无此协议的起终点对照。')
    with st.expander('原始训练数据与诊断'):
        st.dataframe(data,hide_index=True,width='stretch')
        st.json(rows[-1].get('learner',{}))
    with st.expander('协议与证据边界'):
        st.write('官方500配置步，实际499 transition；奖励不修改，五名Blue同一团队回报只展示一次。动作mask与官方合法槽相交；多步动作期间只允许Sleep；保留现有特征供critic使用；有限任务终点不bootstrap。history-correct在Remove后保留已有IOC，成功Restore后才重置。')
        st.write('沿用作者从state提取事件、阻断和decoy元数据的观测接口，不冒称纯Blue observation；未加入红队session/感染真值。当前平坦策略从原始动作端到端训练。四个开发场景仅用于固定模型比较，不代表跨训练seed稳定性或大范围泛化。')
view()
