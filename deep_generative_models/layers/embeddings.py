def compute_embedding_size(variable_size: int, min_embedding_size: int, max_embedding_size: int) -> int:
    return max(min_embedding_size, min(max_embedding_size, int(variable_size / 2)))
