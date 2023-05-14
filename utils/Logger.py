from tqdm import tqdm

class Logger:
    def __init__(self, prefix, use_tqdm=False):
        self.log_path = prefix + ".log"
        self.log_file = open(self.log_path, "w")

        self.use_tqdm = use_tqdm

    def log(self, msg):
        if self.use_tqdm:
            tqdm.write(msg)
        else:
            print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()
    
    def save(self):
        self.log_file.close()
        print(f"[info] Log saved to <{self.log_path}>.")

def print_float_list(lst):
    msg = "["
    for x in lst:
        msg += f"{x:.6f}, "
    msg = msg[:-2] + "]"
    return msg