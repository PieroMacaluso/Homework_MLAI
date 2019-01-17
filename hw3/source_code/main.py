import sys


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("$ python main.py <python_script_cnn>")
        print("-- Execute the script of the specified <python_script_cnn> path")
        print("-- STEP 1: NN1.py")
        print("-- STEP 2: CNN2.py")
        print("-- STEP 3: CNN3a.py or CNN3b.py or CNN3c.py")
        print("-- STEP 4: CNN4a.py or CNN4b.py or CNN4c.py")
        print("-- STEP 5: CNN5a.py or CNN5b.py")
        print("-- STEP 6: CNN6.py")
        print("-- STEP 7: plotting2.py or plotting6.py")
        return -1

    exec(open("./" + sys.argv[1]).read(), globals())


if __name__ == "__main__":
    main()
