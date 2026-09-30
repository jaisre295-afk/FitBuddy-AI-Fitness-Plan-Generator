import os
import time

from dotenv import load_dotenv
from google import genai


# Load .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Please add it to your .env file."
    )


# Gemini client
client = genai.Client(api_key=API_KEY)


# Use the model that is currently working
MODELS = [
    "gemini-3.5-flash-lite",
]


def ask_gemini(prompt: str) -> str:
    last_error = None

    for model in MODELS:
        print(f"\nTrying Gemini model: {model}")

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                if response is None:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                text = getattr(response, "text", None)

                if text:
                    print(f"Gemini success: {model}")
                    return text

                raise RuntimeError(
                    "Gemini returned no text."
                )

            except Exception as error:
                last_error = error

                print(
                    f"Gemini error | "
                    f"model={model} | "
                    f"attempt={attempt + 1}/3"
                )
                print(error)

                error_text = str(error)

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                ):
                    wait_time = 2 ** attempt
                    print(
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    continue

                break

    raise RuntimeError(
        f"Gemini API error: {last_error}"
    )


def generate_workout_plan(
    name=None,
    age=None,
    gender=None,
    weight=None,
    height=None,
    goal=None,
    intensity=None,
    workout_days=None,
    workout_location=None,
):

    prompt = f"""
You are FitBuddy, a friendly AI fitness planning assistant.

Create a safe and realistic 7-day fitness plan based on the
user information below.

USER INFORMATION
----------------
Name: {name}
Age: {age}
Gender: {gender}
Weight: {weight}
Height: {height}
Fitness Goal: {goal}
Workout Intensity: {intensity}
Workout Days Per Week: {workout_days}
Workout Location: {workout_location}


REQUIREMENTS
------------
1. Create a simple 7-day fitness schedule.

2. Include suitable rest and recovery days.

3. Keep the activities appropriate for the user's age,
   experience and selected intensity.

4. Focus on general fitness, movement, strength,
   mobility and recovery.

5. Do not recommend extreme exercise.

6. Do not recommend restrictive eating or extreme dieting.

7. Give general healthy nutrition and hydration guidance.

8. Encourage adequate sleep and recovery.

9. If the user mentions pain, injury or another health concern,
   recommend speaking with a parent/guardian and an appropriate
   healthcare professional.

10. Keep the plan simple and easy to understand.


FORMAT
------

DAY 1
Workout:
Exercises:
Duration:
Recovery:

DAY 2
Workout:
Exercises:
Duration:
Recovery:

DAY 3
Workout:
Exercises:
Duration:
Recovery:

DAY 4
Workout:
Exercises:
Duration:
Recovery:

DAY 5
Workout:
Exercises:
Duration:
Recovery:

DAY 6
Workout:
Exercises:
Duration:
Recovery:

DAY 7
Workout:
Exercises:
Duration:
Recovery:


GENERAL NUTRITION GUIDANCE
--------------------------
Give simple healthy food, hydration and meal guidance.


RECOVERY TIPS
-------------
Give simple sleep, rest and recovery tips.


SAFETY NOTE
-----------
Include a short safety reminder.
"""

    return ask_gemini(prompt)