def normalize_device(arg: str | None) -> str | None:
    """Normalize device input to 'cpu' or 'cuda:<idx>'.

    Parameters
    ----------
    arg: str | None
        Device specification from CLI. Accepted values:
        - None -> auto-select later
        - 'cpu' -> 'cpu'
        - 'cuda' -> 'cuda:0'
        - 'cuda:<n>' -> 'cuda:<n>'
        - '<int>' -> 'cuda:<int>' if >= 0 else 'cpu'
        - '-1' -> 'cpu'
    Returns
    -------
    str | None
        Normalized device string or None for auto-selection.
    """
    if arg is None:
        return None
    s = str(arg).strip()
    if not s:
        return None
    low = s.lower()
    if low == 'cpu' or low == '-1':
        return 'cpu'
    if low == 'cuda':
        return 'cuda:0'
    if low.startswith('cuda:'):
        return f"cuda:{low.split(':', 1)[1]}"
    try:
        idx = int(low)
        return 'cpu' if idx < 0 else f'cuda:{idx}'
    except ValueError:
        return s
