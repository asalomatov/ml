import subprocess


def run_in_shell(cmd, return_output=False):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        if return_output:
            return(stdout)
        else:
            return(0)
    else:
        print(stderr, file=sys.stderr)
        return(1)

if __name__ == "__main__":
    pass