
# BACKWARD CHAINING EXPERT SYSTEM

rules = [
    (["python", "mathematics"], "software_engineering"),
    (["mathematics", "problem_solving"], "data_science"),
    (["data_science", "communication"], "data_analyst"),
    (["python", "hardware_interest"], "embedded_systems"),
    (["software_engineering", "communication"], "technical_lead")
]


def backward_chaining(goal, facts, rules, depth=0):

    space = " " * depth
    print(space + "Goal:", goal)

    if goal in facts:
        print(space + "-> known fact. TRUE")
        return True

    for conditions, conclusion in rules:

        if conclusion == goal:

            print(space + "-> rule found:", conditions, "->", conclusion)

            all_true = True

            for c in conditions:
                if backward_chaining(c, facts, rules, depth + 1) == False:
                    all_true = False

            if all_true:
                print(space + "-> PROVED", goal)
                return True

    print(space + "-> cannot prove", goal)
    return False


# USER INPUT
user_input = input("Enter your skills (comma separated): ")

facts = []

for skill in user_input.split(","):
    facts.append(skill.strip().lower())

goal = input("Enter the goal to prove: ").strip().lower()

# BACKWARD CHAINING
print()
print("--- BACKWARD CHAINING TRACE ---")

answer = backward_chaining(goal, facts, rules)

# OUTPUT
print()
print("RESULT: The goal '" + goal + "' is", answer)
