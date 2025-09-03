from bot_trade.__version__ import __version__, __build_meta__


def main() -> None:
    print(f"[VERSION] {__version__} build={__build_meta__}")


if __name__ == "__main__":
    main()
