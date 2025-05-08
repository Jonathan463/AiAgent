from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import pandas as pd


openai_key = os.getenv("OPENAI_APIKEY")

llm_name ="gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

df = pd.read_csv("./data/name_of_csv_file").fillna(value=0)

print(df.head())

from langchain_experimental.agents.agent_toolkits import (
 create_pandas_dataframe_agent,
 create_csv_agent
)

agent = create_pandas_dataframe_agent(
    llm=model,
    df=df,
    verbose=True,
    #avoid warning for external code import
    allow_dangerous_code=True
)

res = agent.invoke("What is the average salary for the company")
print(res)

#Pre and Sufix Prompt
CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names,then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect onithe answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result, reflect and try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS*, as part of your "Final Answer",
explain how you got to the answer on a section that starts with: "\n\nExplanation: \n".
In the explanation, mention the column names that you used to get to the final answer.
"""

QUESTION= "list all employees whose salary is above mean of the department mean"

# res = agent.invoke(CSV_PROMPT_PREFIX + QUESTION + CSV_PROMPT_SUFFIX)
# print(res)

import streamlit as st

st.title("Database AI with LangChain")
st.write(df.head())

st.write("### Ask a Question")
question = st.text_input(
    "Enter your question about the dataset",
    "Which grade has the highest average base salary, and compare the average female and male average salary"
)

if st.button("Run Query"):
    QUERY = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
    res = agent.invoke(QUERY)
    st.write("### Final Answer")
    st.markdown(res["output"])

