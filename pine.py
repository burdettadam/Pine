from lib import pine
import os
import configparser

def main():
    if not os.path.exists("config.ini"): # Check for config
        raise Exception("config.ini not found!")
    # Clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    config = configparser.SafeConfigParser()
    config.read('config.ini')
    print('''
    ====================================
    Pine: Neural-Network Aimbot (v0.1)
    ====================================

    [INFO] press '0' to quit or ctrl+C in console...''')
    pine.start(config)

if __name__ == '__main__':
    main()


