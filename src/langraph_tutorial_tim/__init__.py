"""
Package entrypoint for langraph-tutorial-tim
"""

from main import run_chatbot


def main() -> None:
    """Entrypoint that executes the main multi-agent chatbot."""
    run_chatbot()


if __name__ == "__main__":
    main()
