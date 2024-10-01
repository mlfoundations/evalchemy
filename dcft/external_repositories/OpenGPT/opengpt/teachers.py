import openai

def ask_openai(prompt, config):
    response = openai.ChatCompletion.create(
        model = config.teacher.model,
        messages = [
            {"role": "user", "content": prompt},
        ]
    )

    message = None
    if response['choices'][0]['finish_reason'] == 'stop':
        message = response['choices'][0]['message']['content']

    return message