def summary() -> None:

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    result =  {
        'Peterson' : 0.764,
        'Schnabel' : 0.102,
        'CAPTURE' : 0.765
    }

    print("\n")
    print(f"{HEADER}MARK-EVALUATE RESULTS{ENDC}")
    print("\n\n")
    print(f"{OKCYAN}Interpretation{ENDC}: 0 poor quality <-> 1 good quality")
    print("\n")
    print(f"Peterson: {OKBLUE}{result['Peterson']}{ENDC}")
    print(f"Schnabel: {OKBLUE}{result['Schnabel']}{ENDC} *")
    print(f"CAPTURE:  {OKBLUE}{result['CAPTURE']}{ENDC}")
    print("\n\n")
    print("* Can be used to assess quality and diversity.")
    print("Use (ref, cand) for quality and (cand, ref) for diversity.")
    print(f"{BOLD}(Mordido, Meinel, 2020){ENDC}")
    print(f"{BOLD}https://arxiv.org/abs/2010.04606{ENDC}")
    print("\n")

summary()