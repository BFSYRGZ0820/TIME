def construct_context(requests_list, tokenizer=None, model=None, task_family=None, device=None):
    texts = []
    for reqs in requests_list:
        # 兼容两种返回格式：
        # 1) 旧版: reqs 是 list/tuple，元素是 request instance
        # 2) 新版: reqs 直接是 request instance
        req = reqs
        if isinstance(reqs, (list, tuple)):
            if len(reqs) == 0:
                continue
            req = reqs[0]

        if hasattr(req, "args") and len(req.args) > 0:
            context_str = req.args[0]
        elif isinstance(req, dict) and "args" in req and len(req["args"]) > 0:
            context_str = req["args"][0]
        else:
            raise TypeError(f"Unsupported request object in construct_context: {type(req)}")

        texts.append(context_str)
    return texts