from bot_trade.run_rl_agent import main
from bot_trade.config.encoding import force_utf8

if __name__ == "__main__":
    import multiprocessing as mp

    force_utf8()
    mp.freeze_support()
    main()
