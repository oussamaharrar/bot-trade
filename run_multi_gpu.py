from bot_trade.run_multi_gpu import main

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
