"""工具入口准备：固定本地数据，仅接收项目名；尚未注册到 LangChain。"""

from project_tasks import select_project_tasks
from task_snapshot import LEARNING_SNAPSHOT


def get_project_tasks(project: str) -> dict:
    """查询 agent 或 bishe 的固定教学任务快照，返回项目、日期、数据性质及任务。"""
    result = select_project_tasks(LEARNING_SNAPSHOT,project)
    return result
