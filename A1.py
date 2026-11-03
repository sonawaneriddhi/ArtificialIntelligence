
# ==========================================================
# PRACTICAL 1 - SMART HOME AGENT
# Simple Reflex Agent
# ==========================================================

# ---------- AGENT FUNCTION ----------

def smart_home_agent(temperature, motion, light, owner_home):

    actions = {}

    # Rule 1: Cooling
    if temperature > 30:
        actions["Fan"] = "ON"
    else:
        actions["Fan"] = "OFF"

    # Rule 2: Lighting
    if motion == "yes" and light == "dark":
        actions["Light"] = "ON"
    else:
        actions["Light"] = "OFF"

    # Rule 3: Security
    if motion == "yes" and owner_home == "no":
        actions["Alarm"] = "ON"
    else:
        actions["Alarm"] = "OFF"

    return actions


# ---------- SENSORS ----------

print("--- SENSOR READINGS ---")

temperature = int(input("Temperature sensor (Celsius): "))
motion = input("Motion sensor (Yes/No): ").lower()
light = input("Light sensor (Light/Dark): ").lower()
owner_home = input("Owner home (Yes/No): ").lower()


# ---------- AGENT ----------

decisions = smart_home_agent(
    temperature,
    motion,
    light,
    owner_home
)


# ---------- ACTUATORS ----------

print("\n---- ACTUATOR STATUS ----")

for device in decisions:
    print(device, "=", decisions[device])
    