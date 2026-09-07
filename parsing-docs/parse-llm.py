from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from parsing-docs/.env into the environment

from openai import OpenAI

client = OpenAI()

file = client.files.create(
    file=open("sample_data/sample_data.pdf", "rb"),
    purpose="user_data"
)

model = "gpt-5.1"

completion = client.chat.completions.create(
    model= model,
    messages=[
        {
            "role":"user",
            "content": [ 
                {
                    "type":"file",
                    "file": {
                        "file_id": file.id,
                    }
                },
                {
                    "type":"text",
                    "text": "Extract the text content of this document. Exclude texts from tables or images.",
                },
            ]
        }
    ]
)

print(completion.choices[0].message.content)

# extract tables
completion = client.chat.completions.create(
    model= model,
    messages=[
        {
            "role":"user",
            "content": [ 
                {
                    "type":"file",
                    "file": {
                        "file_id": file.id,
                    }
                },
                {
                    "type":"text",
                    "text": "Extract the tables from this document. Return in Markdown tables.",},
            ]
        }
    ]
)

print(completion.choices[0].message.content)


import base64

with open("extracted_image_0_0.png", "rb") as f:
   b64 = base64.b64encode(f.read()).decode()

completion = client.chat.completions.create(
      model=model,
      messages=[{
          "role": "user",
          "content": [
              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
              {"type": "text", "text": "Describe this image in detail."},
          ],
      }],
  )
print(completion.choices[0].message.content)