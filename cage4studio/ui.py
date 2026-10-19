"""Streamlit UI, served only on localhost of the Mac Studio.

The Streamlit session's debug env and the training subprocess are separate.
Training topology comes from the run's snapshots, never from the debug env.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from cage4studio.engine import BLUE_NAMES, RED_NAMES, action_name, blue_reward, create_env
from cage4studio.runs import ROOT, dump_json_atomic, list_runs, new_run_dir, read_json, recent_jsonl, status
from cage4studio.topology import dot_graph, snapshot

st.set_page_config(page_title='CAGE4 Studio', page_icon='🛡️', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {padding-top: 1.4rem; max-width: 1550px;}
div[data-testid="stMetric"] {background:rgba(120,150,170,.08); border-radius:12px; padding:13px;}
</style>''', unsafe_allow_html=True)


def fmt_run(r: Path):
    s = status(r)
    return f'{r.name} · {s.get("state", "待启动")}'


def read_runs_selection(key: str):
    runs = list_runs()
    if not runs:
        return None
    chosen = st.selectbox('实验记录', runs, format_func=fmt_run, key=f'selected_run_{key}')
    return chosen


def start_training(c):
    # Single-user local server; avoid second simultaneous run as default.
    running = [r for r in list_runs() if status(r).get('state') == 'running'
               and not (r / 'stop.request').exists()]
    if running:
        st.error('已有训练任务处于运行状态。请先停止它，或等待结束。')
        return
    r = new_run_dir()
    dump_json_atomic(r / 'config.json', c)
    dump_json_atomic(r / 'status.json', {'state': 'starting', 'episodes_total': c['episodes']})
    with (r / 'train.log').open('ab') as log:
        p = subprocess.Popen([sys.executable, '-m', 'cage4studio.train',
                              '--run-dir', str(r), '--config', str(r / 'config.json')],
                             cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    dump_json_atomic(r / 'process.json', {'pid': p.pid})
    st.session_state.selected_run_training = r
    st.success(f'已提交训练任务：{r.name}。刷新后查看运行状态。')


def request_stop(r: Path):
    (r / 'stop.request').touch(exist_ok=True)
    st.info('已请求安全停止，训练进程将在下一仿真步检查到请求。')


def render_graph(snap, key):
    if not snap:
        st.info('还没有网络快照。')
        return
    hosts = snap.get('hosts', [])
    subs = sorted({h['subnet'] for h in hosts})
    overview = st.toggle('仅显示互联网节点与路由器骨干链路', value=True, key=f'overview_{key}')
    zone = '全部'
    if not overview:
        zone = st.selectbox('过滤网段', ['全部'] + subs, key=f'zone_{key}')
    a, b, c, d = st.columns(4)
    a.metric('主机节点', len(hosts))
    b.metric('网络链路', len(snap.get('links', [])))
    c.metric('含红队 Session 的主机', sum(h['red_sessions'] > 0 for h in hosts))
    d.metric('任务阶段', snap.get('mission_phase', '?'))
    st.graphviz_chart(dot_graph(snap, zone, overview), use_container_width=True)
    st.caption('线条来自 CAGE 4 主机 interface.data_links；红色为有 Red Session 的节点。'
               '连接不等于当前可通信性；真实状态只供研究者查看，不喂给学习策略。')
    if st.checkbox('显示主机表及红蓝 Session', value=False, key=f'hosts_{key}'):
        st.dataframe(pd.DataFrame([{k: v for k, v in h.items() if k != 'links'}
                                   for h in hosts]), use_container_width=True, hide_index=True)


st.title('🛡️ CAGE4 Studio')
st.caption('Mac Studio · CybORG / CAGE Challenge 4 · 本地 SSH 隧道访问 · 非官方研究工具')
with st.sidebar:
    st.header('实验配置')
    seed = st.number_input('随机种子', min_value=0, max_value=2**32 - 1, value=7629, step=1)
    steps = st.slider('每轮最大仿真步数', 10, 1000, 100, 10)
    red = st.selectbox('红队策略', RED_NAMES)
    episodes = st.number_input('训练轮数 Episodes', min_value=1, max_value=100000, value=30, step=10)
    lr = st.number_input('学习率', min_value=0.000001, max_value=0.01, value=0.0003,
                         step=0.0001, format='%.6f')
    gamma = st.slider('折扣因子 γ', 0.80, 1.0, 0.99, 0.01)
    device = st.selectbox('模型计算设备', ['cpu', 'mps'], index=0)
    snap_every = st.number_input('每 N 步保存网络快照', min_value=1, max_value=1000, value=10)
    checkpoint_every = st.number_input('每 N 轮保存模型', min_value=1, max_value=100000, value=5)
    conf = {
        'seed': int(seed), 'steps': int(steps), 'red': red, 'episodes': int(episodes),
        'lr': float(lr), 'gamma': float(gamma), 'entropy_coef': 0.01,
        'device': device, 'snapshot_every': int(snap_every),
        'checkpoint_every': int(checkpoint_every),
    }
    st.caption('CPU 为兼容性默认值；MPS 仅加速支持的 PyTorch 运算，仿真仍主要在 CPU。')
    st.download_button('导出当前配置', json.dumps(conf, ensure_ascii=False, indent=2),
                       file_name='cage4_config.json', mime='application/json', use_container_width=True)

training_tab, topology_tab, debug_tab, archive_tab = st.tabs(
    ['📈 训练监控', '🌐 训练中的网络拓扑', '🔬 单步仿真调试', '🗂️ 运行记录与说明'])

with training_tab:
    st.subheader('五蓝队智能体独立 Actor-Critic 基线')
    st.caption('真实策略采样和反向传播，不是模拟曲线；学习结果需与随机策略和 SleepAgent 进行多 Seed 对照，不能预设收敛。')
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button('▶ 开始训练', type='primary', use_container_width=True):
            start_training(conf)
    with c2:
        st.info('改变侧边栏参数不会影响已启动实验。每个 Run 的配置、日志、快照和 checkpoint 独立保存。')

    @st.fragment(run_every='4s')
    def training_monitor():
        r = read_runs_selection('training')
        if not r:
            st.write('当前没有训练记录。点击「开始训练」。')
            return
        s = status(r)
        a, b, c, d = st.columns(4)
        a.metric('状态', s.get('state', '未知'))
        b.metric('轮数', f'{s.get("episode", 0)} / {s.get("episodes_total", "?")}')
        c.metric('当前轮 Step', s.get('step', 0))
        ret = s.get('last_episode_return')
        d.metric('最近一轮 Team Return', '—' if ret is None else f'{ret:.2f}')
        if s.get('state') in ('running', 'starting'):
            if st.button('■ 请求停止训练', key='stop_training'):
                request_stop(r)
        if s.get('state') == 'error':
            st.error(f'训练出错：{s.get("error")}')
            with st.expander('异常栈'):
                st.code(s.get('traceback', '检查 train.log'))
        metrics = recent_jsonl(r / 'metrics.jsonl', 5000)
        if metrics:
            data = pd.DataFrame(metrics)
            st.subheader('团队 Episode Return（非逐步奖励）')
            st.line_chart(data.set_index('episode')['team_return'])
            st.subheader('各 Blue Agent 回报')
            per_agent = pd.DataFrame([{'episode': m['episode'], **m['agents']} for m in metrics])
            st.line_chart(per_agent.set_index('episode'))
            st.dataframe(data[['episode', 'steps', 'team_return', 'total_steps']].tail(25),
                         hide_index=True, use_container_width=True)
        else:
            st.caption('首轮 Episode 训练完后显示奖励曲线，实时网络状态会在右侧标签页按快照频率更新。')
        st.caption(f'TensorBoard：本地 SSH 隧道建立后访问 http://127.0.0.1:6006；'
                   f'服务器保存路径：{r / "tensorboard"}')

    training_monitor()

with topology_tab:
    st.subheader('训练过程网络态势 · Ground Truth（研究者专用）')
    @st.fragment(run_every='4s')
    def live_topology():
        r = read_runs_selection('topology')
        if r is None:
            st.info('先开启一次训练，才能读取真实网络状态快照。')
            return
        snap = read_json(r / 'topology_latest.json')
        if snap:
            mode = st.radio('态势查看方式', ['实时最新', '本 Episode 历史回放'],
                            horizontal=True, key='topology_mode')
            if mode == '本 Episode 历史回放':
                frames = []
                for p in sorted((r / 'snapshots').glob('latest_step_*.json')):
                    recorded = read_json(p)
                    if recorded and recorded.get('episode') == snap['episode']:
                        frames.append((int(recorded['step']), p))
                if frames:
                    step_selected = st.select_slider('保存的 Step 快照',
                                                     options=[s for s, _ in frames],
                                                     value=frames[-1][0], key='replay_step')
                    snap = read_json(dict(frames)[step_selected])
            st.caption(f'Run：{r.name} · Episode {snap["episode"]} / Step {snap["step"]}')
        render_graph(snap, key='training_graph')
        events = recent_jsonl(r / 'actions.jsonl', 150)
        if events:
            st.subheader('近期智能体动作（最新 20 步）')
            st.dataframe(pd.DataFrame(events[-20:][::-1]),
                         hide_index=True, use_container_width=True)
    live_topology()

with debug_tab:
    st.subheader('独立的 SleepAgent 对抗基线 · 可单步推进')
    st.caption('调试仿真与训练进程相互独立。这里的奖励不代表训练策略成绩。')
    if st.button('创建 / 重置单步仿真', use_container_width=False):
        for k in ('debug_env', 'debug_snap', 'debug_trace', 'debug_step'):
            st.session_state.pop(k, None)
    if 'debug_env' not in st.session_state:
        try:
            e = create_env(seed=int(seed), steps=int(steps), red=red)
            e.reset()
            st.session_state.debug_env = e
            st.session_state.debug_snap = snapshot(e, episode=0, step=0)
            st.session_state.debug_step = 0
            st.session_state.debug_trace = []
            st.session_state.debug_config = {'seed': int(seed), 'steps': int(steps), 'red': red}
        except Exception as exc:
            st.error(f'CybORG 加载失败：{exc}')
            st.stop()
    e = st.session_state.debug_env
    cfg = st.session_state.debug_config
    p1, p2, p3 = st.columns(3)
    p1.metric('步骤', f'{st.session_state.debug_step} / {cfg["steps"]}')
    count = 1 if p2.button('推进 1 步') else (10 if p3.button('推进 10 步') else 0)
    if count:
        for _ in range(min(count, cfg['steps'] - st.session_state.debug_step)):
            e.step()
            st.session_state.debug_step += 1
            st.session_state.debug_trace.append({
                'step': st.session_state.debug_step,
                'blue_team_step_reward': blue_reward(e),
                'blue_0_action': action_name(e, 'blue_agent_0'),
                'red_0_action': action_name(e, 'red_agent_0'),
            })
        st.session_state.debug_snap = snapshot(e, episode=0, step=st.session_state.debug_step)
    render_graph(st.session_state.debug_snap, key='debug_graph')
    trace = st.session_state.debug_trace
    if trace:
        st.line_chart(pd.DataFrame(trace).set_index('step')['blue_team_step_reward'])
        st.dataframe(pd.DataFrame(trace[-40:]), hide_index=True, use_container_width=True)
    with st.expander('选取智能体，查看官方 Observation（非 Ground Truth）'):
        who = st.selectbox('智能体', list(BLUE_NAMES) + ['red_agent_0'])
        st.json(json.loads(json.dumps(e.get_observation(who), default=str)))

with archive_tab:
    st.subheader('运行归档')
    r = read_runs_selection('archive')
    if r:
        st.json(read_json(r / 'config.json', {}))
        st.write('运行目录：', str(r))
        st.write('模型权重：', len(list((r / 'checkpoints').glob('*.pt'))), '份')
        st.write('近期训练日志：')
        try:
            content = (r / 'train.log').read_text(encoding='utf-8')[-12000:]
        except FileNotFoundError:
            content = '尚无日志'
        st.code(content, language='text')
        st.download_button('导出本轮指标 JSONL',
                           data=(r / 'metrics.jsonl').read_bytes() if (r / 'metrics.jsonl').exists() else b'',
                           file_name=f'{r.name}_metrics.jsonl')
    st.warning('安全边界：该应用仅应监听 Mac Studio 127.0.0.1 并通过 SSH 转发访问。'
               '训练输出和 Ground Truth 包含敏感实验数据，不建议上传到公共服务。')
    st.caption('CAGE4 Studio 0.1.4 · 实验基线；未集成 RLlib、分布式训练、完整回放或正式 CAGE 评测。')
