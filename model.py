import subprocess

subprocess.check_call(["pip", "install", "numpy==1.16.0"])

if '__main__' == __name__:
    print("Hello world!")