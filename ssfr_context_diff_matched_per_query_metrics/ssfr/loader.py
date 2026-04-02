from typing import Dict, Optional
from .utils import safe_read_text, truncate_text


def load_file_content(file_record: Dict, max_chars: Optional[int] = None) -> str:
    content = safe_read_text(file_record["path"])
    if max_chars is None:
        return content
    return truncate_text(content, max_chars=max_chars)