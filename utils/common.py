def to_list(value):
    if value == None:
        return []
    return value if isinstance(value, list) else [value]
