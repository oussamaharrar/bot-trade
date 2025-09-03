from bot_trade.run_multi_gpu import main
from bot_trade.tools.encoding import force_utf8

if __name__ == "__main__":
    import multiprocessing as mp

    force_utf8()
    mp.freeze_support()
    main()
