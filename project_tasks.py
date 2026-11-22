"""本地只读项目任务查询：业务函数学习单元。"""


def select_project_tasks(snapshot: dict, project: str) -> dict:
    """查询 agent/bishe；未知项目抛出 ValueError，且不修改 snapshot。"""

    if project not in ("agent","bishe"):
        raise ValueError("输入只支持agent和bishe")
    
    projects = snapshot["projects"]
    select_project = projects[project]
    tasks = select_project["tasks"]

    return {
        "project": project,
        "as_of": snapshot["as_of"],
        "data_kind": snapshot["data_kind"],
        "tasks": tasks,
    }





