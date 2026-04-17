"""ECS Task entrypoint — routes to the correct task based on TASK_TYPE env var."""

import os
import sys

if __name__ == "__main__":
    task_type = os.environ.get("TASK_TYPE", "")

    if task_type == "extract_texts":
        from extract_texts import main
        main()
    elif task_type == "write_embeddings":
        from write_embeddings import main
        main()
    else:
        print(f"Unknown TASK_TYPE: '{task_type}'. Must be 'extract_texts' or 'write_embeddings'.")
        sys.exit(1)
