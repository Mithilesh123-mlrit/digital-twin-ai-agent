from agent import agent


print("\n" + "=" * 60)
print("              MY DIGITAL TWIN")
print("=" * 60)
print("Ask me about my skills, projects, experience,")
print("or ask me to calculate something or search the web.")
print("Type 'exit' to quit.")
print("=" * 60)


while True:

    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\nDigital Twin: Goodbye! 👋")
        break

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            }
        )

        final_message = result["messages"][-1]

        print("\nDigital Twin:", final_message.content)

    except Exception as e:

        print("\nDigital Twin: Sorry, something went wrong.")
        print("Error:", e)