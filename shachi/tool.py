def tool(func):
    """Декоратор-маркер, ничего не делает, кроме пометки."""
    func._is_tool = True
    return func