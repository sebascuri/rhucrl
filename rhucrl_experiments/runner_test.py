"""Python Script Template."""


if __name__ == "__main__":
    import os
    import time
    import socket

    print("here")
    try:
        os.mkdir("results/")
    except OSError:
        pass

    with open(f"results/{socket.gethostname()}.txt", "a+") as f:
        f.write(f"{time.time()}\n")
    print(f"saved at time {time.time()} on socket {socket.gethostname()}")
