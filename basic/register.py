
tasks = {}


def register_task(cls):
    task_name = cls.__name__.split(".")[-1]
    if task_name not in tasks:
        tasks[task_name] = cls


def find_task(name):
    task = tasks.get(name, 0)
    if task == 0:
        raise LookupError("can't find this {} task".format(name))
    return task