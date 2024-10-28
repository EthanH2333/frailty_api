# main.py

import os
import json
import argparse

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import load_summarize_chain
from langchain.schema import Document
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --------- Function to Generate Care Plan ---------

def generate_frailty_care_plan(
    first_gait_test_speed: float,
    first_gait_test_time: float,
    first_tug_test_time: float,
    gait_speed_test_risk: str,
    second_gait_test_speed: float,
    second_gait_test_time: float,
    second_tug_test_time: float,
    tug_test_risk: str,
    prisma7: list,  # List of dictionaries with 'question' and 'answer' keys
    uid: str
):
    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Connect to Pinecone vector store
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
    index_name = os.environ.get("INDEX_NAME")

    # Initialize Pinecone
    import pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    vectorstore = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    # Create the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Create the chat model
    chat = ChatOpenAI(temperature=0, model="gpt-4")

    # Create the prompts
    first_invocation_prompt = PromptTemplate(
        input_variables=["input", "context"],
        template="""
You are an expert chatbot focused on frailty care, analyzing a patient's condition based on their PRISMA-7 survey responses and test results. Your task is to provide a factual analysis based solely on the given information. Do not make assumptions or infer information that is not explicitly stated.

Patient's PRISMA-7 Responses and GAIT/TUG Test Results:
<input>
{input}
</input>

Relevant context from the knowledge base:
<context>
{context}
</context>

Please provide a detailed analysis considering only the information provided above. Address the following points:

1. Frailty status: For each response from Patient's PRISMA-7 Responses and GAIT/TUG Test Results, think about how it contributes to the patient's frailty status.
2. Overall frailty assessment: Summarize what the given responses and test results indicate about the patient's frailty status.
3. Key areas of concern: Identify the most critical aspects that need addressing, based solely on the provided information.
4. Potential risks: Discuss potential risks that are directly related to the information given.
5. Care needs: Suggest interventions or support strategies that are relevant to the specific issues mentioned in the input.
6. Interrelations: Explain how the different aspects of the patient's condition, as described in the input, may impact each other.

In your analysis:
- Be specific and refer only to the information provided in the input and context.
- If the input or context doesn't provide sufficient information for any point, clearly state this lack of information.
- Do not make assumptions or infer details that are not explicitly stated.
- If you use information from the context, cite the source.

Your goal is to provide an accurate understanding of the patient's frailty status based strictly on the given information. If there are gaps in the information or if more assessment is needed, state this clearly.

Remember, do not provide any medical advice. Your role is to analyze the given information to support the development of a care plan by healthcare professionals.
"""
    )

    second_invocation_prompt = PromptTemplate(
        input_variables=["input", "analysis", "example"],
        template="""
You are an expert chatbot focused on frailty care, tasked with creating a comprehensive, personalized care plan. Your goal is to synthesize the provided analysis into an actionable, tailored care plan that supports both the caretaker and the frailty patient.

You avoid humor or casual language due to the seriousness of the topic.

You are provided the following information and analysis of the patient's condition.
Patient's PRISMA-7 Responses, and Gait and TUG Test results:
<input>
{input}
</input>

I have conducted the following analysis of the patient's condition:
<analysis>
{analysis}
</analysis>

Based on this analysis, create a comprehensive care plan that addresses the specific needs and circumstances of this frailty patient.

First, you must begin your care plan by summarizing all the responses from the PRISMA-7 survey, and the Gait and TUG test results.
Next, continue by saying "As a caretaker, you should consider the following:".
Then, the care plan should:

1. Provide a concise summary of the patient's overall frailty status, highlighting key areas of concern.

2. Outline 4-5 key care recommendations. For each recommendation:
   a) Clearly state the recommendation
   b) Explain the rationale behind it, citing specific aspects of the patient's condition
   c) Provide detailed, practical steps for implementation
   d) Identify potential challenges and suggest strategies to overcome them

3. Address safety considerations specific to this patient's situation, including both home safety and broader health and wellbeing measures.

4. Suggest a monitoring and evaluation plan to track the patient's progress and adjust care as needed.

5. Recommend specific resources or support services that would be particularly beneficial for this patient.

6. Identify any areas where additional assessment or professional consultation might be necessary, explaining why.

Throughout your care plan:
- Ensure each recommendation is clearly linked to specific aspects of the patient's condition.
- Prioritize interventions that address the most critical aspects of the patient's frailty status.
- Consider the interplay between physical, cognitive, and social aspects of the patient's health.
- Include both short-term interventions for immediate concerns and long-term strategies for ongoing care.
- Provide clear, actionable guidance that can be readily implemented by caregivers.

Your care plan should be comprehensive, practical, and tailored to both the patient's needs and the caretaker's ability to implement it.

If there are any uncertainties or gaps in your knowledge, please say so and do not make up information. Clearly state what additional information or next steps would be required from healthcare providers.

Your care plan should be comprehensive yet practical, providing clear guidance that can be readily implemented by caregivers while also serving as a valuable resource for healthcare professionals involved in the patient's care.

Remember, your plan should be tailored to the patient's needs, and also meaningful to help caretakers as well.

While knowledgeable about frailty care, you stay within your role of developing a care plan to support the caretaker and frailty patient, without providing definitive medical advice. Should there be any uncertainty, you should state this, and suggest the user to speak with a licensed healthcare professional.

Here is an example format of a care plan:
<example>
{example}
</example>
"""
    )

    # Create the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": first_invocation_prompt}
    )

    # Prepare the PRISMA-7 data
    prisma7_dict = {
        item['question']: 'Yes' if item['answer'] else 'No'
        for item in prisma7
    }

    # Prepare the input data
    input_data = {
        **prisma7_dict,
        "First Gait Test speed": f"{first_gait_test_speed} meters per second (m/s).",
        "First Gait Test time": f"{first_gait_test_time} seconds",
        "First TUG Test time": f"{first_tug_test_time} seconds",
        "Gait Speed Test Risk": gait_speed_test_risk,
        "Second Gait Test speed": f"{second_gait_test_speed} meters per second (m/s).",
        "Second Gait Test time": f"{second_gait_test_time} seconds",
        "Second TUG Test time": f"{second_tug_test_time} seconds",
        "TUG Test Risk": tug_test_risk,
    }

    # Run the first invocation
    first_result = qa({"query": str(input_data)})

    # Run the second invocation
    final_care_plan = chat([
        {
            "role": "system",
            "content": second_invocation_prompt.format(
                input=str(input_data),
                analysis=first_result["result"],
                example="Your example care plan here"
            )
        }
    ])

    # Format the care plan as a big string
    care_plan_string = f"""
Care Plan:
{final_care_plan.content}

Sources used:
{chr(10).join(f"{i+1}. {doc.metadata.get('source', 'Unknown')}" for i, doc in enumerate(first_result['source_documents']))}
"""

    # Save the care plan to a file
    output_directory = '/home/ubuntu/UserPlan'
    os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists

    output_file_path = os.path.join(output_directory, f'{uid}.txt')

    with open(output_file_path, 'w') as file:
        file.write(care_plan_string)

    # Return the care plan string
    return care_plan_string

# Add this function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate Frailty Care Plan')
    parser.add_argument('--input_data', type=str, required=True, help='Input data as JSON string')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    input_data = json.loads(args.input_data)

    # Extract variables from input_data
    prisma7 = input_data.get('prisma7', [])
    uid = input_data.get('uid', 'default_uid')

    result = generate_frailty_care_plan(
        first_gait_test_speed=input_data.get('First Gait Test speed', 0.0),
        first_gait_test_time=input_data.get('First Gait Test time', 0.0),
        first_tug_test_time=input_data.get('First TUG Test time', 0.0),
        gait_speed_test_risk=input_data.get('Gait Speed Test Risk', ''),
        second_gait_test_speed=input_data.get('Second Gait Test speed', 0.0),
        second_gait_test_time=input_data.get('Second Gait Test time', 0.0),
        second_tug_test_time=input_data.get('Second TUG Test time', 0.0),
        tug_test_risk=input_data.get('TUG Test Risk', ''),
        prisma7=prisma7,
        uid=uid
    )

    print("Care plan generated successfully.")
