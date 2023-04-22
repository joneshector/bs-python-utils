from bs_python_utils.bs_logging import init_logger, log_execution


@log_execution
def in_and_out():
    print("in, and out")
    return


def fib(n):
    print(f"{n}=n")
    if n > 2:
        logger.info("recursing")
        return fib(n - 1) + fib(n - 2)
    else:
        logger.warning("done")
        return 1


if __name__ == "__main__":
    logger_dir = "logs"
    logger_name = "test_log"
    logger = init_logger(logger_name, save_dir=logger_dir)
    logger.info("main starting")
    m = fib(5)
    in_and_out()
