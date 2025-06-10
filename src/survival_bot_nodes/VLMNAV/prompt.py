def construct_action_prompt(goal: str, num_actions: int, turn_around_available: bool):
    """
    Constructs the action selection prompt for the VLM.
    """
    
    turn_around_note = "NOTE: Only choose action 0 if you absolutely cannot move toward the goal or no viable actions are possible. Otherwise, always choose the best action to reach the target." if turn_around_available else ""
    prompt = (
        f"You are an embodied robotic assistant with an RGB image sensor. "
        f"Your task is to navigate to the nearest {goal.upper()}, and get as close to it as possible. "
        f"Use your prior knowledge about where items are typically located within a home. "
        f"There are {num_actions} red lines superimposed onto your observation, which represent potential actions. "
        f"These are labeled with numbers in white circles, representing the locations you would move to if you took that action. "
        f"{turn_around_note}"
        f"First, describe what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. "
        f"Second, determine which general direction you should go in. "
        f"Lastly, explain which action best achieves that, and **return your answer in the exact format {{'action': <action_number>}}**. "
        f"Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS."
    )
    return prompt

if __name__ == "__main__":
    goal = "volleyball"
    num_actions = 5
    turn_around_available = True
    prompt = construct_action_prompt(goal, num_actions, turn_around_available)
    print(prompt)